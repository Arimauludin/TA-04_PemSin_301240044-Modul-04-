# 📈 Peramalan Harga Penutupan Harian Bitcoin (BTC) Menggunakan LSTM

Aplikasi web interaktif berbasis **Deep Learning (Long Short-Term Memory)** untuk memprediksi pergerakan harga penutupan harian (*Close Price*) aset *cryptocurrency* Bitcoin (BTC) berdasarkan data historis deret waktu (*time series*).

Proyek ini disusun untuk memenuhi **Tugas Akhir 04 (TA-04) — Modul 04: Long Short-Term Memory (LSTM)** pada Mata Kuliah Praktikum Pembelajaran Mesin, Program Studi Teknik Informatika, Universitas Bale Bandung.

---

## 👤 Identitas Mahasiswa
* **Nama:** Ari Mauludin
* **NIM:** 301240044
* **Kelas:** 5C — Teknik Informatika
* **Institusi:** Universitas Bale Bandung
* **Dosen Pengampu:** Mohammad Bayu Anggara, S.Kom., M.Kom.

---

## ✨ Fitur Utama
* **Pemodelan Time Series Deep Learning:** Menggunakan arsitektur *Sequential* 2-layer LSTM untuk mempelajari dependensi pola harga historis selama 60 hari (*timesteps* = 60).
* **Evaluasi Metrik Regresi:** Menampilkan performa model pada data uji yang telah dikembalikan ke skala asli USD menggunakan metrik **RMSE** (*Root Mean Squared Error*) dan **MAE** (*Mean Absolute Error*).
* **Visualisasi Interaktif (Chart.js):** Grafik perbandingan kurva harga aktual dengan hasil prediksi model serta garis proyeksi masa depan.
* **Simulasi Horizon Peramalan (1–30 Hari):** Pengguna dapat mengatur *slider* untuk memproyeksikan estimasi harga Bitcoin hingga 30 hari ke depan secara rekursif.
* **Antarmuka 3D Glassmorphism:** Dilengkapi latar belakang jaringan partikel 3D interaktif menggunakan *Three.js* dan kartu efek miring 3D (*Vanilla-Tilt.js*).

---

## 📊 Dataset & Arsitektur Model

### Dataset
* **Sumber:** [Cryptocurrency Historical Prices — Kaggle](https://www.kaggle.com/datasets/sudalairajkumar/cryptocurrencypricehistory) (`coin_Bitcoin.csv`)
* **Kolom Waktu (*Timestamp*):** `Date`
* **Variabel Target:** `Close` (Harga penutupan harian dalam USD)
* **Pembagian Data:** 80% Data Latih (*Training*) dan 20% Data Uji (*Testing*) secara kronologis.

### Arsitektur Jaringan Saraf (Sequential LSTM)
```text
Input (60, 1)
   │
   ▼
LSTM (50 units, return_sequences=True)
   │
   ▼
Dropout (0.2)
   │
   ▼
LSTM (50 units, return_sequences=False)
   │
   ▼
Dropout (0.2)
   │
   ▼
Dense (25 units)
   │
   ▼
Dense (1 unit) ──► Prediksi Harga Penutupan (USD)