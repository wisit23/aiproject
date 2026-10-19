"""Image restoration demo: resize -> add noise/blur -> run selected model -> show results."""
import base64, importlib.util, io, os, sys

# oneDNN makes this TensorFlow build ~100x slower on CPU
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image

BASE = Path(__file__).parent
MODELS_DIR = BASE / "models"      # one entry-point file per selectable model
WEIGHTS_DIR = BASE / "weights"    # checkpoints (.pth / .h5)
sys.path.insert(0, str(BASE))     # lets models/*.py do `from archs.x import ...`
app = Flask(__name__)
_cache = {}


# ---------- degradation (same as the notebook) ----------
def add_noise(img, sigma, seed):
    if sigma <= 0:
        return img
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def add_blur(img, ksize, sigma):
    if sigma <= 0:
        return img
    ksize = int(ksize) | 1  # must be odd
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)


# ---------- models ----------
def list_models():
    """Any .keras/.h5 file, or a .py file with predict(img_float01) [and optional load()]."""
    items = [p for p in MODELS_DIR.iterdir() if p.suffix in {".keras", ".h5", ".py"}]
    return sorted(p.name for p in items if not p.name.startswith("_"))


def get_model(name):
    if name in _cache:
        return _cache[name]
    path = MODELS_DIR / name
    if path.suffix == ".py":
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "load"):
            mod.load(WEIGHTS_DIR)
        fn = mod.predict
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(path, compile=False)
        tile = model.input_shape[1] or None

        def fn(x):
            return run_tiled(model, x, tile)
    _cache[name] = fn
    return fn


def run_tiled(model, x, tile):
    """Model was trained on 128x128; bigger images are split into 50%-overlapping
    tiles and blended with a Hann window so tile seams disappear."""
    h, w = x.shape[:2]
    if not tile or (h == tile and w == tile):
        return model(x[None], training=False).numpy()[0]
    stride, m = tile // 2, tile // 4
    # margin so image borders aren't at the low-weight edge of a window
    xp = np.pad(x, ((m, m), (m, m), (0, 0)), mode="reflect")
    H, W = xp.shape[:2]
    ph = max(tile - H, 0) + (-(max(H, tile) - tile)) % stride
    pw = max(tile - W, 0) + (-(max(W, tile) - tile)) % stride
    xp = np.pad(xp, ((0, ph), (0, pw), (0, 0)), mode="reflect")
    H, W = xp.shape[:2]

    win1 = np.hanning(tile + 2)[1:-1].astype(np.float32)  # strictly > 0
    win = (win1[:, None] * win1[None, :])[:, :, None]

    pos = [(i, j) for i in range(0, H - tile + 1, stride)
                  for j in range(0, W - tile + 1, stride)]
    out = np.zeros_like(xp)
    wsum = np.zeros(xp.shape[:2] + (1,), np.float32)
    for k in range(0, len(pos), 32):  # batch to cap memory
        chunk = pos[k:k + 32]
        preds = model(np.stack([xp[i:i + tile, j:j + tile] for i, j in chunk]),
                      training=False).numpy()
        for (i, j), p in zip(chunk, preds):
            out[i:i + tile, j:j + tile] += p * win
            wsum[i:i + tile, j:j + tile] += win
    out /= wsum
    return out[m:m + h, m:m + w]


# ---------- helpers ----------
def to_b64(img):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    m = ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))
    return float(m.mean())


def metrics(ref, img):
    p = psnr(ref, img)
    return {"psnr": None if p == float("inf") else round(p, 2), "ssim": round(ssim(ref, img), 4)}


# ---------- routes ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def api_models():
    return jsonify(list_models())


@app.post("/api/process")
def api_process():
    f = request.files["image"]
    size = int(request.form.get("size", 128))
    noise = float(request.form.get("noise", 0))
    blur_sigma = float(request.form.get("blur_sigma", 0))
    blur_k = int(request.form.get("blur_k", 7))
    seed = int(request.form.get("seed", 42))
    model_name = request.form.get("model", "")

    img = Image.open(f.stream).convert("RGB")
    # center-crop to square, then resize
    w, h = img.size; s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    clean = np.array(img.resize((size, size), Image.BICUBIC))

    degraded = add_noise(add_blur(clean, blur_k, blur_sigma), noise, seed)
    result = {"clean": to_b64(clean), "degraded": to_b64(degraded),
              "degraded_metrics": metrics(clean, degraded)}

    if model_name:
        try:
            x = degraded.astype(np.float32) / 255.0
            y = np.asarray(get_model(model_name)(x), dtype=np.float32)
            restored = (np.clip(y, 0, 1) * 255).round().astype(np.uint8)
            result["restored"] = to_b64(restored)
            result["restored_metrics"] = metrics(clean, restored)
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
