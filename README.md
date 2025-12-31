# Sistem Deteksi Lubang Jalan Berbasis Deep Learning

## Deskripsi Proyek
Proyek UAS Visi Komputer yang mengimplementasikan sistem deteksi lubang jalan (pothole) secara otomatis menggunakan model YOLOv11. Sistem ini dapat mendeteksi dan melokalisasi kerusakan jalan berupa lubang dari gambar atau video.

### Informasi Mahasiswa
- **Nama**: M. Taris Rizki
- **Mata Kuliah**: Visi Komputer (S1 Informatika USK)
- **Semester**: Ganjil 2025/2026

## Latar Belakang
Kondisi jalan yang rusak dan berlubang merupakan permasalahan infrastruktur yang sering ditemui di berbagai daerah. Lubang jalan dapat menyebabkan:
- Kecelakaan lalu lintas
- Kerusakan kendaraan
- Kemacetan akibat pengemudi menghindari lubang

Dengan memanfaatkan teknologi Computer Vision dan Deep Learning, sistem ini dapat membantu mengidentifikasi lokasi lubang jalan secara otomatis untuk mempercepat proses perbaikan infrastruktur.

## Teknologi yang Digunakan
- **Python 3.11**
- **Ultralytics YOLOv11** - Model object detection state-of-the-art
- **OpenCV** - Image processing
- **Jupyter Notebook** - Development environment

## Struktur Dataset
```
├── train/
│   ├── images/     # 54 gambar training
│   └── labels/     # Anotasi YOLO format
├── val/
│   ├── images/     # 10 gambar validasi
│   └── labels/     # Anotasi YOLO format
├── test/
│   └── images/     # 9 gambar testing
└── data.yaml       # Konfigurasi dataset
```

## Hasil Training
| Metrik | Nilai |
|--------|-------|
| Epochs | 30 |
| mAP50 | 80.8% |
| Precision | 79.7% |
| Recall | 69.2% |
| mAP50-95 | 42.8% |

## Cara Penggunaan

### 1. Instalasi Dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### 2. Training Model
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=30, imgsz=640)
```

### 3. Inference/Deteksi
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train_new/weights/best.pt')
results = model.predict(source='test/images', save=True, conf=0.25)
```

## Contoh Hasil Deteksi
Model berhasil mendeteksi lubang jalan pada gambar test dengan confidence score yang tinggi. Hasil deteksi tersimpan di folder `output_detection/results/`.

## Referensi
1. Ultralytics YOLOv11 Documentation
2. Szeliski, Richard. Computer Vision: Algorithms and Applications. Springer, 2010.
3. Howse, Joseph. Learning OpenCV 4 Computer Vision with Python 3. Packt Publishing, 2020.

## Lisensi
Proyek ini dibuat untuk keperluan akademik mata kuliah Visi Komputer USK.
