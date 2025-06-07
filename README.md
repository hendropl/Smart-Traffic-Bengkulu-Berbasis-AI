# Smart-Traffic-Bengkulu-Berbasis-AI

# Smart Traffic Bengkulu Berbasis AI

Proyek ini bertujuan untuk mengembangkan sistem pengelolaan lalu lintas cerdas untuk Kota Bengkulu menggunakan teknologi Artificial Intelligence (AI). Sistem ini mampu memprediksi kemacetan, memberikan rekomendasi pengaturan lampu lalu lintas, dan menampilkan data visual secara real-time untuk membantu pengambilan keputusan oleh pemerintah kota.

---

## 🚦 Fitur Utama

- Prediksi kondisi lalu lintas berbasis data historis dan real-time.
- Visualisasi data lalu lintas melalui dashboard interaktif.
- Rekomendasi penyesuaian pengaturan lampu lalu lintas.
- Integrasi data dari CCTV, sensor, dan API eksternal (cuaca, lalu lintas).
- Basis pengembangan sistem Smart City masa depan.

---

## 🧠 Model AI yang Digunakan

- **LSTM (Long Short-Term Memory):** digunakan untuk prediksi volume lalu lintas berdasarkan data sekuensial.
- **Decision Tree / Random Forest:** untuk klasifikasi kepadatan lalu lintas (macet/lancar/sedang).
- **Reinforcement Learning (opsional):** digunakan pada tahap lanjutan untuk mengontrol durasi lampu lalu lintas secara adaptif berdasarkan reward waktu tempuh minimum.

---

## 📊 Jenis dan Sumber Data

- **Volume kendaraan:** dari CCTV atau API Waze/Google Maps.
- **Waktu dan lokasi:** timestamp dan koordinat GPS.
- **Cuaca:** dari API BMKG.
- **Hari khusus:** data hari libur nasional, akhir pekan, dll.

### Metode Pengumpulan & Praproses:

- Scraping atau API call berkala.
- Pembersihan data (handling missing/null).
- Normalisasi dan ekstraksi fitur waktu (jam, hari, musim).
- Labeling otomatis atau semi-manual kondisi lalu lintas.

---

## 🛠️ Instalasi dan Menjalankan Proyek

1. Clone repositori:
   ```bash
   git clone https://github.com/hendropl/Smart-Traffic-Bengkulu-Berbasis-AI.git
   cd Smart-Traffic-Bengkulu-Berbasis-AI
