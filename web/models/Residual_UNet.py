"""Residual U-Net (TensorFlow/Keras) trained on noise + Gaussian blur."""
from archs.residual_unet import build_unet

model = None


def load(weights_dir):
    global model
    model = build_unet()
    model.load_weights(str(weights_dir / "residual_unet.weights.h5"))


def predict(x):
    # fully convolutional: any size divisible by 8 works
    return model(x[None], training=False).numpy()[0]
