# Prediksi Harga Harian Komoditas Pangan Strategis dengan Machine Learning

Proyek ML berbasis **SDG 2 (Tanpa Kelaparan)**, **Target 2.c**: memperkirakan harga eceran harian komoditas pangan strategis (cabai rawit merah, cadangan beras medium) dari data historis, dengan tiga model: baseline, advanced, dan deep learning.

> **Status:** Minggu 1 [L] Learning Problem (menunggu Checkpoint 1).
> Bagian bertanda `[ ... ]` diisi setelah dataset dicek.

## Informasi Proyek

| Komponen | Isian |
|---|---|
| Nama / NIM | Muhammad Raufa Hafid Widodo/244311052 & Raihan Firdaus Alfaritsi/244311055 |
| Program Studi | [ D4 - TRPL ] |
| Mata Kuliah |  Data Science  |
| Dosen Pengampu | Gus Nanang Syaifuddiin  |
| Tahun Akademik | 2026/2027 |
| SDG | SDG 2: Tanpa Kelaparan |
| Target SDG | Target 2.c (indikator 2.c.1: indikator anomali harga pangan) |
| Link Proposal | [ [...](https://docs.google.com/document/d/1CpmrUSJtKF-bzT2bKn79GwzQz6rLENhJYhV4S2uIjug/edit?tab=t.3chb62tavuk4#heading=h.h8q3abq7kmms) ] |
| Link Video Pembahasan | [ ... ] |

## 1. Ringkasan

Harga pangan seperti cabai dan beras sering berubah tajam akibat musim, cuaca, dan distribusi. Proyek ini membangun model forecasting time series untuk memperkirakan **harga hari berikutnya**, lalu membandingkan tiga pendekatan model dengan protokol evaluasi yang ketat (split kronologis, tanpa data leakage, test set dipakai sekali).

Hasil model diposisikan sebagai **perkiraan**, bukan kepastian, sehingga dapat membantu informasi pasar jangka pendek bagi konsumen, pedagang, dan pemerintah daerah.

## 2. Keterkaitan dengan SDGs

| Komponen | Isi |
|---|---|
| SDG | SDG 2: Tanpa Kelaparan |
| Target | Target 2.c: menjamin pasar komoditas pangan berfungsi baik dan informasi pasar tersedia tepat waktu untuk membatasi volatilitas harga pangan yang ekstrem |
| Masalah nyata | Lonjakan harga cabai dan beras merugikan rumah tangga berpenghasilan rendah, petani dan pedagang kecil, serta pemerintah daerah |
| Peran ML | Mempelajari tren, musiman, dan ketergantungan antar-hari untuk memperkirakan harga berikutnya |
| Kontribusi | Tidak langsung, lewat informasi harga yang lebih terukur |

## 3. Learning Problem

| Komponen | Isi |
|---|---|
| **Target** | Harga eceran harian cabai rawit merah (Rp/kg) pada hari berikutnya. Regresi / forecasting time series satu langkah ke depan |
| **Objective** | Meminimalkan MSE. Metrik utama: MAE. Pelengkap: RMSE dan MAPE. Pembanding: prediksi naif (harga kemarin) |
| **Data** | Time series harian dari PIHPS/WFP: [jumlah baris, rentang tanggal, wilayah]. Data berurutan waktu dan tidak i.i.d. |

## 4. Dataset

| Sumber | Tautan | Catatan |
|---|---|---|
| PIHPS Nasional (Bank Indonesia) | https://www.bi.go.id/hargapangan | Sumber resmi harga pangan strategis. Pencacahan pada hari kerja, jadi akhir pekan kosong |
| WFP Indonesia Food Prices (HDX) | https://data.humdata.org/dataset/wfp-food-prices-for-indonesia | CSV, periode 2007 sampai Mei 2024, sumber BPS dan PIHPS. Cek granularitas harian atau bulanan |
| Kaggle | [https://data.humdata.org/dataset/wfp-food-prices-for-indonesia](https://www.kaggle.com/datasets/muhyusuf1112/indonesia-commodity-price-based-piphs-source) | CSV, harga ritel harian dari 10 komoditas pangan strategis di 34 provinsi di Indonesia , yang diambil dari Pusat Informasi Harga Pangan Strategis (PIHPS) Bank Indonesia - Pusat Informasi Harga Pangan Strategis Nasional resmi. Mencakup periode 1 Januari 2022 hingga 12 Februari 2026 (~1.460 hari), dataset ini menyediakan lebih dari 500.000 titik data untuk ketahanan pangan, analisis inflasi, dan penelitian ekonomi. |

Dataset **tidak diunggah** ke repository (lihat `.gitignore`). Unduh sendiri dan simpan di `data/raw/`.

Cek ukuran dan rentang data (sesuaikan nama file dan kolom):

```python
import pandas as pd

df = pd.read_csv("data/raw/nama_file.csv")
print(df.shape)
df.info()
print(df.head())

# contoh: rentang tanggal dan nilai hilang (sesuaikan nama kolom)
# df["date"] = pd.to_datetime(df["date"], errors="coerce")
# print(df["date"].min(), df["date"].max())
# print(df.isna().sum())
```

## 5. Rencana Tiga Model

| Model | Pilihan | Alasan awal |
|---|---|---|
| 1. Baseline | Linear Regression pada lag harga (1, 2, 7 hari). Pembanding: prediksi naif | Sederhana dan mudah dijelaskan |
| 2. Advanced | XGBoost pada lag, statistik bergulir, hari dalam minggu, dan bulan | Menangkap hubungan non-linear dan musiman dari fitur tabular |
| 3. Deep learning | LSTM pada jendela 14 sampai 30 hari, minimal 10 epoch | Mempelajari ketergantungan sekuensial tanpa fitur lag manual |

## 6. Alur Lifecycle (L-M-O-P)

- **[L] Learning Problem:** rumusan masalah, dataset, EDA, data preparation.
- **[M] Model:** desain baseline, XGBoost, dan LSTM beserta hypothesis space dan inductive bias.
- **[O] Optimization:** tuning, early stopping, validasi walk-forward, diagnosis training.
- **[P] Performance:** evaluasi di test set, perbandingan model, analisis kesalahan, iterasi.

## 7. Struktur Repository

```
.
├── data/
│   ├── raw/                 # dataset asli (tidak di-commit)
│   └── processed/           # data hasil preprocessing
├── notebooks/
│   ├── 01_learning_problem_eda.ipynb        # [L]
│   ├── 02_data_preparation.ipynb            # [L]
│   ├── 03_model_design_baseline_ml.ipynb    # [M]
│   ├── 04_deep_learning_optimization.ipynb  # [M+O]
│   └── 05_evaluation_comparison.ipynb       # [P]
├── src/                     # script modular
├── models/                  # model tersimpan
├── reports/
│   └── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

## 8. Cara Menjalankan

1. Clone repository dan masuk ke foldernya.
2. Buat environment dan pasang dependensi:

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Unduh dataset ke `data/raw/` (lihat bagian 4).
4. Jalankan notebook berurutan dari `01` sampai `05`. Notebook juga dapat dijalankan di Google Colab atau Kaggle.

Library utama yang direncanakan: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `xgboost`, dan `tensorflow` atau `torch`. Versi persisnya dicatat di `requirements.txt` setelah environment final.

**Reproducibility:**
- Random seed 42 di semua library.
- Notebook dapat dijalankan ulang dari awal sampai akhir.
- Test set tidak dipakai saat training atau tuning.
- Versi library tercatat.

## 9. Pertimbangan Etika

- **Salah prediksi** dapat memicu keputusan merugikan (penimbunan, panic buying), jadi hasil diposisikan sebagai perkiraan.
- **Bias wilayah dan periode:** daerah atau periode dengan data sedikit bisa kurang terwakili. Laporkan performa per periode.
- **Lonjakan tak terduga** (cuaca ekstrem, kebijakan) sulit diprediksi dan akan dibahas dalam analisis kesalahan.
- **Privasi:** data harga bersifat agregat, risiko privasi rendah.

## 10. Status Pengerjaan

**Minggu 1 [L]**
- [ ] Target dan jenis tugas ML jelas
- [ ] Objective dan metrik utama ditentukan
- [ ] Dataset dapat diakses dan ukurannya dicek
- [ ] Minimal 2 referensi terkumpul
- [ ] Ide berasal dari mahasiswa dan selaras dengan SDG 2 (Target 2.c)
- [ ] Peran ML dan kontribusi terhadap SDG dijelaskan
- [ ] Pertimbangan etika dicatat
- [ ] Repository dibuat dan dosen ditambahkan sebagai collaborator
- [ ] Ide, SDG, dan dataset disetujui dosen (Checkpoint 1)

**Minggu berikutnya**
- [ ] Data understanding dan preparation
- [ ] Baseline dan XGBoost
- [ ] LSTM (minimal 10 epoch, plot loss)
- [ ] Evaluasi dan perbandingan model
- [ ] Laporan akhir dan video

## 11. Referensi

1. Amirudin, Prasetyo, S. A., & Triono, G. (2025). *[Judul lengkap, verifikasi dari dokumen asli]*. Universitas Budi Luhur.
2. [Penulis]. (2026). *[Judul lengkap, verifikasi dari dokumen asli]*. Jurnal Informatika Terpadu, 12(1).
3. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
4. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780.
6. United Nations. (n.d.). *Transforming our world: The 2030 Agenda for Sustainable Development*. https://sdgs.un.org/2030agenda
