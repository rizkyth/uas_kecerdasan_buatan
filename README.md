# UAS Kecerdasan Buatan

## Prediksi Kelulusan Siswa Menggunakan Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

# Dataset https://www.kaggle.com/datasets/devansodariya/student-performance-data?resource=download

## 📋 Informasi Proyek

| Item            | Detail                                                          |
| --------------- | --------------------------------------------------------------- |
| **Judul**       | Prediksi Kelulusan Siswa Menggunakan Algoritma Machine Learning |
| **Nama**        | Mahasiswa                                                       |
| **Mata Kuliah** | Kecerdasan Buatan                                               |
| **Dataset**     | Student Performance Dataset (Kaggle/UCI)                        |

---

## 📁 Struktur Repository

```
UAS-KecerdasanBuatan/
├── README.md                    ← Dokumen ini
├── Laporan_uas.md               ← Laporan lengkap UAS
├── uas_model.py                 ← Script Python utama (full pipeline)
└── data/
    ├── student-performance.csv  ← Dataset
    ├── eda_visualization.png    ← Grafik EDA
    ├── confusion_matrix.png     ← Confusion matrix semua model
    ├── model_comparison.png     ← Perbandingan metrik model
    ├── model_summary.csv        ← Ringkasan hasil evaluasi
    └── Jurnal/                  ← Referensi jurnal ilmiah
```

---

## 🎯 Tujuan Proyek

Membangun model Machine Learning untuk memprediksi apakah seorang siswa akan **lulus** atau **tidak lulus** berdasarkan faktor akademik dan sosial, menggunakan dataset Student Performance dari Kaggle.

---

## 🔧 Langkah Pengerjaan

### Step 1: Persiapan Lingkungan

```bash
# Install dependensi yang diperlukan
pip install pandas numpy scikit-learn matplotlib seaborn
```

Dependensi yang digunakan:

- `pandas` — manipulasi dan analisis data
- `numpy` — komputasi numerik
- `scikit-learn` — algoritma Machine Learning
- `matplotlib` & `seaborn` — visualisasi data

---

### Step 2: Load dan Eksplorasi Dataset

Dataset Student Performance berisi data 395 siswa dari dua sekolah menengah di Portugal dengan 33 fitur meliputi:

- Informasi demografis (usia, jenis kelamin, alamat)
- Latar belakang keluarga (pendidikan orang tua, pekerjaan)
- Perilaku akademik (waktu belajar, absensi, kegagalan masa lalu)
- Nilai ujian (G1, G2, G3)

**Target prediksi**: `lulus` (1 jika G3 ≥ 10, 0 jika G3 < 10)

---

### Step 3: Exploratory Data Analysis (EDA)

1. **Visualisasi distribusi** nilai akhir (G3) menggunakan histogram
2. **Distribusi kelas** kelulusan (bar chart)
3. **Analisis korelasi** antar fitur menggunakan heatmap
4. **Scatter plot** G1 vs G3 untuk melihat hubungan nilai antar semester
5. **Deteksi imbalanced class**: rasio 60:40 (lulus:tidak lulus)

---

### Step 4: Data Preparation

1. **Pembersihan data**: Hapus duplikat dan missing values (tidak ada dalam dataset ini)
2. **Label Encoding**: Konversi semua fitur kategorikal (17 kolom) ke numerik
3. **Standardisasi**: Terapkan `StandardScaler` untuk normalisasi fitur numerik
4. **Split data**: 80% training, 20% testing (stratified)

---

### Step 5: Modeling (4 Algoritma)

| Algoritma     | Parameter Utama       |
| ------------- | --------------------- |
| Decision Tree | `max_depth=5`         |
| KNN           | `n_neighbors=7`       |
| SVM           | `kernel='rbf', C=1.0` |
| Naive Bayes   | Default (GaussianNB)  |

---

### Step 6: Evaluasi

Evaluasi menggunakan metrik:

- **Accuracy** — proporsi prediksi benar
- **Precision** — ketepatan prediksi positif
- **Recall** — kemampuan mendeteksi kelas positif
- **F1-Score** — harmonik rata-rata precision & recall (metrik utama)

**Confusion matrix** divisualisasikan untuk setiap model.

---

### Step 7: Analisis Hasil dan Kesimpulan

Model terbaik dipilih berdasarkan F1-Score tertinggi:

| Algoritma     | Accuracy   | F1-Score   |
| ------------- | ---------- | ---------- |
| Decision Tree | 0.8101     | 0.8315     |
| KNN           | 0.7595     | 0.8081     |
| **SVM** ⭐    | **0.7975** | **0.8367** |
| Naive Bayes   | 0.7975     | 0.8298     |

**Model Terbaik: SVM** dengan F1-Score 0.8367

---

## 🚀 Cara Menjalankan

```bash
# 1. Clone repository
git clone https://github.com/[username]/UAS-KecerdasanBuatan.git
cd UAS-KecerdasanBuatan

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan pipeline lengkap
python uas_model.py
```

---

## 📊 Hasil Utama

- ✅ 4 algoritma berhasil diimplementasikan dan dibandingkan
- ✅ SVM terpilih sebagai model terbaik (F1: 0.84)
- ✅ G1, G2, dan failures adalah fitur paling berpengaruh
- ✅ Visualisasi EDA dan confusion matrix tersedia

---

## 📚 Referensi Jurnal

1. Cortez & Silva (2008) — Using Data Mining to Predict Secondary School Student Performance
2. Romero & Ventura (2010) — Educational Data Mining: A Review
3. Baker & Inventado (2014) — Educational Data Mining and Learning Analytics
4. Pedregosa et al. (2011) — Scikit-learn: Machine Learning in Python
5. Han, Kamber & Pei (2011) — Data Mining: Concepts and Techniques

---

_UAS Kecerdasan Buatan | Semester Genap 2025/2026_
