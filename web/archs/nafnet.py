"""NAFNet (Chen et al., ECCV 2022) - inference-only port of megvii-research/NAFNet.

Parameter names match the official checkpoints, so their .pth files load with strict=True.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + self.eps).sqrt()
        return self.weight.view(1, -1, 1, 1) * y + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        # simplified channel attention
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.sg = SimpleGate()

        ffn = ffn_expand * c
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp):
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        y = inp + self.conv3(x) * self.beta

        x = self.sg(self.conv4(self.norm2(y)))
        return y + self.conv5(x) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=(), dec_blk_nums=()):
        super().__init__()
        self.intro = nn.Conv2d(img_channel, width, 3, padding=1)
        self.ending = nn.Conv2d(width, img_channel, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.ups, self.downs = nn.ModuleList(), nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, 2 * chan, 2, 2))
            chan *= 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        H, W = inp.shape[2:]
        ph, pw = -H % self.padder_size, -W % self.padder_size
        inp = F.pad(inp, (0, pw, 0, ph))

        x = self.intro(inp)
        encs = []
        for enc, down in zip(self.encoders, self.downs):
            x = enc(x)
            encs.append(x)
            x = down(x)
        x = self.middle_blks(x)
        for dec, up, skip in zip(self.decoders, self.ups, encs[::-1]):
            x = dec(up(x) + skip)
        x = self.ending(x) + inp
        return x[:, :, :H, :W]


# ---- helpers shared by the web model files ----
def load_nafnet(weights_path, **cfg):
    model = NAFNet(**cfg)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt.get("params", ckpt))
    return model.eval()


def predict_nafnet(model, x):
    """x: HxWx3 float32 in [0,1] -> same shape."""
    t = torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None]
    with torch.no_grad():
        y = model(t)[0].permute(1, 2, 0).numpy()
    return y
