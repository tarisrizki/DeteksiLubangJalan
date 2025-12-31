# LAPORAN PROYEK AKHIR
# MATA KULIAH VISI KOMPUTER

---

## **SISTEM DETEKSI LUBANG JALAN BERBASIS DEEP LEARNING MENGGUNAKAN ARSITEKTUR YOLOv11**

---

**Disusun Oleh:**

**M. Taris Rizki**

---

**PROGRAM STUDI S1 INFORMATIKA**
**FAKULTAS MATEMATIKA DAN ILMU PENGETAHUAN ALAM**
**UNIVERSITAS SYIAH KUALA**
**BANDA ACEH**
**2025**

---

# DAFTAR ISI

1. [Latar Belakang](#1-latar-belakang)
2. [Tujuan dan Manfaat](#2-tujuan-dan-manfaat)
3. [Solusi dan Inovasi yang Diusulkan](#3-solusi-dan-inovasi-yang-diusulkan)
4. [Link Video Presentasi](#4-link-video-presentasi)
5. [Referensi](#5-referensi)

---

# 1. LATAR BELAKANG

Infrastruktur jalan merupakan komponen vital dalam sistem transportasi yang menghubungkan berbagai wilayah dan mendukung mobilitas masyarakat. Kondisi jalan yang baik sangat penting untuk menjamin kelancaran arus transportasi serta keselamatan pengguna jalan. Namun, kerusakan jalan berupa lubang (*pothole*) masih menjadi permasalahan serius yang sering ditemui di berbagai daerah di Indonesia, termasuk di Provinsi Aceh.

Lubang jalan terbentuk akibat berbagai faktor, seperti beban kendaraan yang berlebihan, curah hujan tinggi, serta kualitas konstruksi yang kurang memadai. Keberadaan lubang jalan dapat menimbulkan berbagai dampak negatif yang signifikan. Pertama, lubang jalan dapat menyebabkan kecelakaan lalu lintas, terutama bagi pengendara sepeda motor yang rentan kehilangan keseimbangan saat melewati lubang. Kedua, lubang jalan dapat menyebabkan kerusakan pada komponen kendaraan seperti ban, velg, dan sistem suspensi yang memerlukan biaya perbaikan yang tidak sedikit. Ketiga, keberadaan lubang jalan juga dapat menyebabkan kemacetan karena pengemudi cenderung mengurangi kecepatan atau mengubah jalur untuk menghindari lubang.

Proses identifikasi dan pemetaan lokasi lubang jalan secara konvensional memerlukan waktu dan tenaga yang cukup besar. Petugas harus melakukan survei langsung ke lapangan untuk mengidentifikasi lokasi kerusakan jalan satu per satu. Metode ini tentu tidak efisien mengingat panjangnya ruas jalan yang harus dipantau. Oleh karena itu, diperlukan sebuah sistem otomatis yang dapat membantu mengidentifikasi keberadaan lubang jalan secara cepat dan akurat.

Perkembangan teknologi *Computer Vision* dan *Deep Learning* membuka peluang untuk mengembangkan sistem deteksi objek secara otomatis. Arsitektur YOLO (*You Only Look Once*) merupakan salah satu model *object detection* yang populer karena kemampuannya melakukan deteksi secara *real-time* dengan tingkat akurasi yang tinggi. Versi terbaru, YOLOv11, menawarkan peningkatan performa baik dari segi kecepatan inferensi maupun akurasi deteksi dibandingkan versi sebelumnya.

Berdasarkan latar belakang tersebut, proyek ini mengembangkan sebuah sistem deteksi lubang jalan berbasis *deep learning* menggunakan arsitektur YOLOv11. Sistem ini diharapkan dapat membantu proses identifikasi lokasi lubang jalan secara otomatis sehingga dapat mempercepat penanganan perbaikan infrastruktur jalan.

---

# 2. TUJUAN DAN MANFAAT

## 2.1 Tujuan

Proyek ini memiliki beberapa tujuan yang ingin dicapai sebagai berikut.

Tujuan pertama adalah membangun model *object detection* yang mampu mendeteksi dan melokalisasi keberadaan lubang jalan pada gambar atau video. Model yang dikembangkan diharapkan dapat mengidentifikasi lubang jalan dengan tingkat akurasi yang tinggi, ditunjukkan dengan nilai *mean Average Precision* (mAP) yang memuaskan.

Tujuan kedua adalah mengevaluasi performa model deteksi menggunakan berbagai metrik evaluasi standar dalam domain *object detection*. Metrik yang digunakan meliputi *Precision*, *Recall*, mAP50, dan mAP50-95 yang masing-masing memberikan perspektif berbeda mengenai kualitas deteksi model.

Tujuan ketiga adalah mengimplementasikan model yang telah dilatih untuk melakukan inferensi pada gambar baru yang belum pernah dilihat sebelumnya. Kemampuan generalisasi model menjadi aspek penting untuk memastikan sistem dapat digunakan dalam kondisi nyata di lapangan.

## 2.2 Manfaat

Proyek ini memberikan beberapa manfaat yang dapat dirasakan oleh berbagai pihak.

Bagi instansi pemerintah yang menangani infrastruktur jalan, sistem ini dapat membantu mempercepat proses inventarisasi kerusakan jalan. Data lokasi lubang jalan yang terdeteksi dapat dijadikan dasar untuk menyusun prioritas perbaikan sehingga alokasi anggaran dapat dilakukan secara lebih efisien dan tepat sasaran.

Bagi masyarakat pengguna jalan, sistem ini secara tidak langsung dapat meningkatkan keselamatan berkendara. Dengan adanya sistem deteksi otomatis, diharapkan proses perbaikan jalan dapat dilakukan lebih cepat sehingga risiko kecelakaan akibat lubang jalan dapat diminimalisir.

Bagi pengembangan ilmu pengetahuan, proyek ini memberikan kontribusi dalam penerapan teknologi *Computer Vision* untuk menyelesaikan permasalahan nyata di masyarakat. Dokumentasi dan kode program yang dihasilkan dapat menjadi referensi bagi peneliti atau pengembang lain yang ingin mengembangkan sistem serupa.

---

# 3. SOLUSI DAN INOVASI YANG DIUSULKAN

## 3.1 Deskripsi Solusi

Solusi yang diusulkan dalam proyek ini adalah pengembangan sistem deteksi lubang jalan berbasis *deep learning* menggunakan arsitektur YOLOv11. YOLO merupakan arsitektur *object detection* yang bekerja dengan pendekatan *single-shot detection*, yaitu melakukan deteksi dan lokalisasi objek dalam satu kali proses inferensi. Pendekatan ini memungkinkan deteksi dilakukan secara *real-time* dengan kecepatan tinggi tanpa mengorbankan akurasi deteksi.

Sistem dikembangkan menggunakan bahasa pemrograman Python dengan memanfaatkan library Ultralytics yang menyediakan implementasi YOLO yang teroptimasi. Proses training dilakukan menggunakan GPU NVIDIA GeForce RTX 3050 untuk mempercepat komputasi. Dataset yang digunakan terdiri dari gambar-gambar jalan yang memiliki lubang dengan anotasi *bounding box* dalam format YOLO.

## 3.2 Arsitektur Model

Model yang digunakan adalah YOLOv11 varian nano (yolo11n) yang merupakan varian paling ringan namun tetap memberikan performa yang baik. Arsitektur ini terdiri dari 100 layer dengan total 2.582.347 parameter yang dapat dilatih. Varian nano dipilih karena keseimbangan antara kecepatan inferensi dan akurasi yang cocok untuk aplikasi *real-time*.

Proses training dilakukan selama 30 epoch dengan ukuran gambar input 640×640 piksel dan batch size 8. Teknik *early stopping* diterapkan dengan patience 10 epoch untuk mencegah *overfitting*. Hasil training menunjukkan model mencapai mAP50 sebesar 80.8%, *Precision* 79.7%, *Recall* 69.2%, dan mAP50-95 sebesar 42.8%.

## 3.3 Link Repository

Seluruh kode program dan dataset dapat diakses melalui repository GitHub pada tautan berikut:

**🔗 GitHub Repository:**
https://github.com/tarisrizki/DeteksiLubangJalan

Repository tersebut berisi:
- Jupyter Notebook untuk training dan inferensi (`deteksi_lubang_jalan.ipynb`)
- Dataset training, validasi, dan testing
- Model weights hasil training (`runs/detect/deteksi_lubang/weights/best.pt`)
- Dokumentasi lengkap dalam file README

## 3.4 Petunjuk Penggunaan

Berikut adalah langkah-langkah untuk menggunakan sistem deteksi lubang jalan yang telah dikembangkan.

**Langkah 1: Clone Repository**
```bash
git clone https://github.com/tarisrizki/DeteksiLubangJalan.git
cd DeteksiLubangJalan
```

**Langkah 2: Instalasi Dependencies**
```bash
pip install ultralytics opencv-python matplotlib
```

**Langkah 3: Menjalankan Deteksi**
```python
from ultralytics import YOLO

# Load model yang sudah ditraining
model = YOLO('runs/detect/deteksi_lubang/weights/best.pt')

# Deteksi pada gambar
results = model.predict(source='path/to/image.jpg', save=True, conf=0.25)
```

**Langkah 4: (Opsional) Training Ulang**
```python
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(data='data.yaml', epochs=30, imgsz=640)
```

## 3.5 Hasil dan Evaluasi

Model yang dikembangkan berhasil mendeteksi lubang jalan dengan performa sebagai berikut:

| Metrik | Nilai |
|--------|-------|
| mAP50 | 80.8% |
| Precision | 79.7% |
| Recall | 69.2% |
| mAP50-95 | 42.8% |
| Inference Speed | ~7 ms/gambar |

Hasil tersebut menunjukkan bahwa model memiliki kemampuan yang baik dalam mendeteksi lubang jalan. Nilai mAP50 sebesar 80.8% mengindikasikan bahwa sebagian besar lubang jalan dapat terdeteksi dengan benar pada threshold IoU 50%.

---

# 4. LINK VIDEO PRESENTASI

Video presentasi proyek dapat diakses melalui tautan Google Drive berikut:

**🎬 Video Presentasi:**
https://drive.google.com/drive/folders/1EkL1SrnBonhYOdSEPcFCdKQvAx3Qb3ue?usp=sharing

Video tersebut berisi penjelasan mengenai latar belakang proyek, metodologi yang digunakan, demonstrasi sistem, serta pembahasan hasil yang diperoleh.

---

# 5. REFERENSI

[1] Jocher, G., Qiu, J., & Chaurasia, A. (2024). Ultralytics YOLO11. *Ultralytics*. https://docs.ultralytics.com

[2] Wang, C.-Y., Yeh, I.-H., & Liao, H.-Y. M. (2024). YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information. *arXiv preprint arXiv:2402.13616*.

[3] Terven, J., Córdova-Esparza, D. M., & Romero-González, J. A. (2023). A Comprehensive Review of YOLO Architectures in Computer Vision: From YOLOv1 to YOLOv8 and YOLO-NAS. *Machine Learning and Knowledge Extraction*, 5(4), 1680–1716.

[4] Silva, L. A., Sanchez San Blas, H., Peral García, D., Sales Mendes, A., & Villarubia González, G. (2022). An Architectural Multi-Agent System for a Pavement Monitoring System with Pothole Recognition in UAV Images. *Sensors*, 20(21), 6205.

[5] Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., & Sekimoto, Y. (2022). RDD2022: A Multi-National Image Dataset for Automatic Road Damage Detection. *arXiv preprint arXiv:2209.08538*.

[6] Fan, R., Bocus, M. J., Zhu, Y., Jiao, J., Wang, L., Ma, F., ... & Liu, M. (2021). Road Crack Detection Using Deep Convolutional Neural Network and Adaptive Thresholding. *IEEE Intelligent Vehicles Symposium (IV)*, 474-479.

---

**Banda Aceh, 31 Desember 2025**

**Penyusun,**

**M. Taris Rizki**
