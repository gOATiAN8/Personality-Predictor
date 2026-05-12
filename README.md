PERSONALITY PREDICTOR

Personality Predictor adalah aplikasi web berbasis Machine Learning yang dirancang untuk memprediksi tipe kepribadian seseorang secara akurat dan cepat. Proyek ini dibangun dengan fokus pada kemudahan penggunaan, visualisasi data yang informatif, dan implementasi algoritma Random Forest yang stabil.


FITUR UTAMA

- Prediksi Kepribadian Otomatis: Menggunakan variabel perilaku sosial untuk menentukan tipe kepribadian Introvert atau Ekstrovert.

- Visualisasi Data Interaktif: Grafik donut untuk menunjukkan persentase probabilitas hasil prediksi secara visual.

- Analisis Perilaku Sosial: Menganalisis 7 indikator kunci mulai dari interaksi sosial hingga aktivitas media sosial.

- Antarmuka Web Modern: Tampilan responsif dan intuitif menggunakan framework Streamlit.

- Tingkat Akurasi Tinggi: Model telah diuji dan mencapai akurasi sebesar 93.56%.


TEKNOLOGI YANG DIGUNAKAN

- Python 3: Bahasa pemrograman utama untuk logika dan pemrosesan data.

- Streamlit: Framework untuk membangun antarmuka web interaktif berbasis Python.

- Scikit-Learn: Pustaka Machine Learning untuk implementasi algoritma Random Forest.

- Plotly: Library visualisasi data untuk grafik yang interaktif.

- Pandas & Numpy: Digunakan untuk manipulasi dan analisis data numerik.


STRUKTUR PROYEK

- app.py: Pusat logika aplikasi dan pengaturan antarmuka web.

- best_model.pkl: Model Machine Learning yang sudah dilatih.

- scaler.pkl: File untuk normalisasi dan penskalaan data input.

- personality_dataset.csv: Dataset utama yang digunakan untuk pengembangan model.

- requirements.txt: Daftar pustaka Python yang diperlukan untuk menjalankan proyek.


PANDUAN INSTALASI LOKAL

1. Clone Repository
Lakukan clone pada repositori ini menggunakan perintah:
git clone https://github.com/goATIAN8/Personality-Predictor-main.git

2. Instalasi Library
Jalankan perintah berikut pada terminal untuk memasang semua dependensi:
pip install -r requirements.txt

3. Jalankan Aplikasi
Gunakan perintah berikut dan akses aplikasi melalui alamat http://localhost:8501:
streamlit run app.py

