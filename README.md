<p align="center">
  <img src="https://img.shields.io/badge/YOLOv11-Object%20Detection-blue?style=for-the-badge&logo=pytorch" alt="YOLO">
  <img src="https://img.shields.io/badge/Python-3.11-green?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/mAP50-80.8%25-success?style=for-the-badge" alt="mAP50">
</p>

# 🚗 Sistem Deteksi Lubang Jalan Berbasis Deep Learning

Sistem deteksi otomatis untuk mengidentifikasi lubang jalan (*pothole*) menggunakan model **YOLOv11** dengan akurasi tinggi. Proyek ini dikembangkan sebagai tugas akhir mata kuliah **Visi Komputer**.

---

## 📋 Informasi Proyek

| Item | Detail |
|------|--------|
| **Nama** | M. Taris Rizki |
| **Mata Kuliah** | Visi Komputer B |
| **Institusi** | Universitas Syiah Kuala |
| **Semester** | Ganjil 2025/2026 |

---

## 🎯 Latar Belakang

Kondisi jalan yang rusak dan berlubang merupakan permasalahan infrastruktur serius yang dapat menyebabkan:
- 🚨 Kecelakaan lalu lintas
- 🔧 Kerusakan kendaraan
- 🚦 Kemacetan akibat pengemudi menghindari lubang

Dengan memanfaatkan teknologi **Computer Vision** dan **Deep Learning**, sistem ini dapat membantu mengidentifikasi lokasi lubang jalan secara otomatis untuk mempercepat proses perbaikan infrastruktur.

---

## 🛠️ Teknologi yang Digunakan

| Teknologi | Keterangan |
|-----------|------------|
| **Python 3.11** | Bahasa pemrograman utama |
| **Ultralytics YOLOv11** | Model object detection state-of-the-art |
| **OpenCV** | Image processing |
| **PyTorch + CUDA** | GPU acceleration |
| **Jupyter Notebook** | Development environment |

---

## 📁 Struktur Proyek

```
📦 DeteksiLubangJalan
├── 📂 train/
│   ├── 📂 images/          # 54 gambar training
│   └── 📂 labels/          # Anotasi format YOLO
├── 📂 val/
│   ├── 📂 images/          # 10 gambar validasi
│   └── 📂 labels/          # Anotasi format YOLO
├── 📂 test/
│   └── 📂 images/          # 9 gambar testing
├── 📂 runs/detect/
│   └── 📂 deteksi_lubang/  # Hasil training
│       └── 📂 weights/
│           └── best.pt     # Model terbaik
├── 📄 data.yaml            # Konfigurasi dataset
├── 📄 deteksi_lubang_jalan.ipynb  # Notebook utama
└── 📄 README.md
```

---

## 📊 Hasil Training

| Metrik | Nilai | Status |
|--------|-------|--------|
| **Epochs** | 30 | ✅ |
| **mAP50** | **80.8%** | ✅ Bagus |
| **Precision** | **79.7%** | ✅ Akurat |
| **Recall** | **69.2%** | ⚠️ Sedang |
| **mAP50-95** | **42.8%** | ⚠️ Sedang |
| **Inference Speed** | ~7ms/image | ✅ Real-time |

---

## 🚀 Cara Penggunaan

### 1. Clone Repository
```bash
git clone https://github.com/tarisrizki/DeteksiLubangJalan.git
cd DeteksiLubangJalan
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### 3. Training Model (Opsional)
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=30, imgsz=640)
```

### 4. Deteksi Lubang Jalan
```python
from ultralytics import YOLO

# Load model
model = YOLO('runs/detect/deteksi_lubang/weights/best.pt')

# Deteksi pada gambar
results = model.predict(source='test/images', save=True, conf=0.25)
```

---

## 📸 Contoh Hasil Deteksi

Model berhasil mendeteksi lubang jalan pada gambar test dengan bounding box dan confidence score:

| Gambar | Deteksi |
|--------|---------|
| potholes1.jpg | 1 lubang |
| potholes3.jpg | 2 lubang |
| potholes54.jpg | 9 lubang |
| potholes57.jpg | 3 lubang |

---

## 📚 Referensi

1. Ultralytics YOLOv11 Documentation - https://docs.ultralytics.com
2. Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M. (2024). YOLOv9: Learning What You Want to Learn Using Programmable Gradient   Information. arXiv preprint arXiv:2402.13616.
3. Terven, J., Córdova-Esparza, D. M., & Romero-González, J. A. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS. Machine Learning and Knowledge Extraction, 5(4), 1680–1716.

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik mata kuliah **Visi Komputer** di Universitas Syiah Kuala.

---

<p align="center">
  <b>🎓 Proyek UAS Visi Komputer 2025</b><br>
  M. Taris Rizki - Universitas Syiah Kuala
</p>
