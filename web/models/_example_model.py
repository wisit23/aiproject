"""Template for a custom model. Copy, rename (no leading underscore), and edit.

The web app calls:
    load(models_dir)  -> optional, run once (load weights here)
    predict(x)        -> x is HxWx3 float32 in [0,1]; return the same shape in [0,1]
"""
import numpy as np

model = None


def load(models_dir):
    global model
    # import tensorflow as tf
    # model = tf.keras.models.load_model(models_dir / "my_weights.keras", compile=False)


def predict(x):
    if model is None:
        return x  # identity, for testing
    return model(x[None], training=False).numpy()[0]
