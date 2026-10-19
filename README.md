# aiproject — Image Restoration Demo

เว็บเดโม Denoise / Deblur: อัปโหลดรูป → เติม blur + noise → ให้โมเดลกู้คืน → เทียบ PSNR / SSIM

## โครงสร้าง

```
web/
├── app.py              Flask backend (degrade, metrics, โหลดโมเดล)
├── templates/index.html
├── models/             1 ไฟล์ = 1 ตัวเลือกใน dropdown (มี load() และ predict())
│   ├── Residual_UNet.py    TensorFlow
│   ├── NAFNet_SIDD.py      PyTorch, denoise
│   ├── NAFNet_GoPro.py     PyTorch, motion deblur
│   └── _example_model.py   เทมเพลต (ขึ้นต้น _ จึงไม่แสดง)
├── archs/              นิยามสถาปัตยกรรมของโมเดล (ไม่แสดงใน dropdown)
│   ├── residual_unet.py
│   └── nafnet.py
└── weights/            ไฟล์น้ำหนัก (.pth / .h5)
```

## รัน

```bash
cd web
pip install -r requirements.txt
python app.py        # http://localhost:5000
```

## เพิ่มโมเดลใหม่

1. ใส่สถาปัตยกรรมใน `archs/` (ถ้ามี) และวางน้ำหนักใน `weights/`
2. คัดลอก `models/_example_model.py` เปลี่ยนชื่อ (ไม่ขึ้นต้นด้วย `_`) แล้วเขียน
   - `load(weights_dir)` — โหลดน้ำหนักครั้งเดียว
   - `predict(x)` — `x` เป็น HxWx3 float32 ช่วง 0–1, คืนรูปร่างเดียวกัน
3. รีสตาร์ทเว็บ ชื่อไฟล์จะขึ้นใน dropdown เอง

ไฟล์ `.keras` / `.h5` ที่วางตรงๆ ใน `models/` ก็ใช้ได้ (โหลดผ่าน Keras)

## หมายเหตุ

- NAFNet weights มาจาก [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) (width32)
- NAFNet_SIDD ลบ noise ได้ดี ส่วน NAFNet_GoPro เทรนกับ motion blur ไม่ใช่ noise จึงไม่เหมาะกับภาพที่มี noise
- ไฟล์ `.pth` ใหญ่ (~65–110 MB) จึงไม่ได้ commit ไว้ใน git
