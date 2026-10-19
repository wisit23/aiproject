"""NAFNet width32 trained on SIDD (real-camera denoising)."""
from archs.nafnet import load_nafnet, predict_nafnet

model = None


def load(weights_dir):
    global model
    model = load_nafnet(weights_dir / "NAFNet-SIDD-width32.pth", width=32,
                        enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[2, 2, 2, 2])


def predict(x):
    return predict_nafnet(model, x)
