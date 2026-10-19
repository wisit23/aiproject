"""NAFNet width32 trained on GoPro (motion deblurring)."""
from archs.nafnet import load_nafnet, predict_nafnet

model = None


def load(weights_dir):
    global model
    model = load_nafnet(weights_dir / "NAFNet-GoPro-width32.pth", width=32,
                        enc_blk_nums=[1, 1, 1, 28], middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1])


def predict(x):
    return predict_nafnet(model, x)
