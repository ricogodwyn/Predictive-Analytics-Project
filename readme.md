# Laporan Proyek Machine Learning - Frederico Godwyn

## Domain Proyek

Dalam era digital saat ini, pemasaran berbasis data telah menjadi strategi yang sangat efektif untuk meningkatkan penjualan dan mencapai target pasar. Dataset yang kami kembangkan mencakup variabel penting seperti anggaran promosi di TV, media sosial, dan radio, serta kolaborasi dengan berbagai jenis influencer (Mega, Macro, Nano, dan Micro), dan hasil penjualan. Dengan menggunakan dataset ini, saya bertujuan untuk mengembangkan algoritma machine learning yang dapat menganalisis hubungan antara pengeluaran promosi dan penjualan, dan dapat memprediksi sales bedasarkan promosi yang dilakukan.
## Business Understanding
### Problem statements 
1.  Terdapat ketidakpastian mengenai saluran promosi mana yang memberikan dampak terbesar terhadap penjualan.
2. Perusahaan kesulitan dalam menentukan jenis influencer yang paling sesuai untuk kampanye pemasaran mereka.
3. Meskipun perusahaan telah menginvestasikan anggaran yang signifikan dalam promosi, terdapat tantangan dalam mengoptimalkan pengeluaran tersebut untuk mencapai hasil penjualan yang maksimal.
### Goals 
1. Mengembangkan model machine learning yang dapat menganalisis dan mengukur kontribusi masing-masing saluran promosi (TV, media sosial, radio, dan influencer) terhadap penjualan, sehingga perusahaan dapat mengalokasikan anggaran dengan lebih efektif.
2. Membangun analisis yang mengidentifikasi jenis influencer yang paling efektif dalam meningkatkan penjualan berdasarkan data historis, sehingga perusahaan dapat memilih influencer yang tepat untuk kampanye mereka.
3. Mengembangkan algoritma optimasi yang memberikan rekomendasi pengeluaran anggaran promosi yang optimal di berbagai saluran, dengan tujuan memaksimalkan penjualan dan meningkatkan return on investment (ROI) dari kampanye pemasaran.

## Data Understanding
untuk dataset di sini. Dataset yang saya gunakan saya peroleh dari kaggle (https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data/data) Dataset ini berisikan tentang marketing dan sales data. Dan dalam proyek saya, saya menggunakan regresion untuk mencari prediksi penjualan, dari beberapa data lain, data yang terdapat di dalam dataset tersebut adalah:
1. TV promotion budget (in million)
2. Social Media promotion budget (in million)
3. Radio promotion budget (in million)
4. Influencer: Whether the promotion collaborate with Mega, Macro, Nano, Micro influencer
5. Sales (in million)

## Data Preparation
1. **Pemeriksaan Nilai Null (Null Values Check)**:<br>
Deskripsi: Langkah pertama yang dilakukan adalah memeriksa apakah terdapat nilai null dalam dataset. Nilai null dapat muncul akibat kesalahan pengumpulan data atau ketidaklengkapan informasi.
Alasan: Memeriksa nilai null sangat penting karena keberadaan nilai ini dapat mempengaruhi analisis dan hasil model. Data yang memiliki nilai null perlu ditangani, baik dengan menghapus baris tersebut, mengisi dengan nilai rata-rata, median, atau menggunakan teknik imputasi lainnya.
2. **Pemeriksaan Nilai Nol (Zero Values Check)**:<br>
Deskripsi: Setelah memeriksa nilai null, langkah selanjutnya adalah memeriksa sel-sel yang memiliki nilai nol. Nilai nol dalam konteks anggaran promosi atau penjualan dapat menunjukkan bahwa tidak ada investasi atau hasil yang dicapai.
Alasan: Memeriksa nilai nol penting untuk memastikan bahwa data yang digunakan dalam analisis adalah representatif. Nilai nol yang tidak relevan dapat mengganggu analisis dan memberikan hasil yang menyesatkan. Jika nilai nol tidak sesuai dengan konteks, perlu dilakukan penanganan, seperti penghapusan atau penggantian dengan nilai yang lebih relevan.
3. **Pencarian Duplikat** (Duplicate Check):<br>
Deskripsi: Langkah terakhir dalam proses data preparation adalah mencari dan mengidentifikasi duplikat dalam dataset. Duplikat dapat terjadi akibat kesalahan dalam pengumpulan data atau penggabungan dataset.
Alasan: Mengidentifikasi dan menghapus duplikat sangat penting untuk menjaga integritas data. Duplikat dapat menyebabkan bias dalam analisis dan mempengaruhi hasil model, sehingga perlu dihilangkan agar analisis yang dilakukan akurat dan dapat diandalkan.
4. **Split Test - Train & Standarization**
Deskripsi: harus dilakukan split test dan train untuk membagi antara dataset untuk training (feature sales dihapus) dan test untuk mencari tahu seberapa akuratnya data. Selain itu juga dilakukan standarisasi.
Alasan: split test-train dilakukan untuk membagi dataset menjadi 2, yaitu train dan test. Data train digunakan untuk training dan test digunakan untuk mengetahui seberapa akuratnya data, tergantung dengan metrik yang dipakai (saya menggunakan mse). Standarization digunakan pada dataset train sebelum proses training dan digunakan pada dataset test sebelum di test. Hal ini digunakan agar algo ML berfungsi secara maksimal.
5. **Encoding pada categorical feature**
Deskripsi: digunakan encoding pada data categorical, yaitu proses mengganti data categorical dengan value lain seperti angka dan True or false. Saya menggunakan ini pada column influencer.
Alasan: Encoding kategorikal diperlukan karena algoritma machine learning umumnya tidak dapat memproses data dalam format teks. Dengan mengubah variabel kategorikal menjadi format numerik, kita dapat memastikan bahwa model dapat memahami dan memanfaatkan informasi tersebut dalam analisis dan prediksi.
## Modelling
Saya menggunakan 3 jenis algo ML, yaitu:
1. KNN
KNN adalah algoritma yang mengklasifikasikan data berdasarkan kedekatan dengan titik data lainnya. Model ini menghitung jarak antara titik data dan memilih K tetangga terdekat untuk menentukan kelas.
    * Kelebihan:
        1. Sederhana dan mudah dipahami.
        2. Tidak memerlukan asumsi distribusi data.
    * Kekurangan:
        1. Sensitif terhadap skala data dan outlier.
        2. Memerlukan waktu komputasi yang tinggi untuk dataset besar.
2. Random Forest:
Random Forest adalah ensemble learning method yang menggunakan banyak pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dilatih pada subset acak dari data.
    * Kelebihan:
        1. Tahan terhadap overfitting.
        2. Dapat menangani data yang hilang dan variabel kategorikal.
    * Kekurangan:
        1. Model yang lebih kompleks dan sulit untuk diinterpretasikan.
        2. Memerlukan lebih banyak sumber daya komputasi dibandingkan model sederhana.
3. Boosting (misalnya, AdaBoost atau Gradient Boosting):
Boosting adalah teknik ensemble yang menggabungkan beberapa model lemah untuk membentuk model yang kuat. Model ini berfokus pada kesalahan yang dibuat oleh model sebelumnya dan berusaha untuk memperbaikinya.
    * Kelebihan:
        1. Mampu meningkatkan akurasi secara signifikan.
        2. Efektif dalam menangani data yang tidak seimbang.
    * Kekurangan:
        1. Rentan terhadap overfitting jika tidak diatur dengan baik.
        2. Proses pelatihan yang lebih lambat dibandingkan dengan model lain.

Saya sudah mengetes ketiga ini dan mendapat kesimpulan Random Forest lebih akurat daripada yang lain (dilakukan dengan data test)

## Evaluation
Dalam proyek ini, metrik evaluasi yang digunakan untuk menilai performa model adalah Mean Squared Error (MSE). MSE mengukur rata-rata kuadrat dari selisih antara nilai yang diprediksi oleh model dan nilai aktual. Metrik ini memberikan gambaran yang jelas tentang seberapa baik model dalam memprediksi hasil, dengan penalti yang lebih besar untuk kesalahan yang lebih besar. Semakin rendah nilai MSE, semakin baik performa model.
Hasil Evaluasi Berdasarkan MSE
Berikut adalah hasil MSE untuk masing-masing model pada dataset pelatihan (train) dan pengujian (test):

| Model| MSE (Train) | MSE (Test) | 
| ---- | ----------- |----------- |
| KNN  |0.066774  | 0.087716 |
| Random Forest (RF)| 0.001631  |0.01067|
| Boosting|0.121198|0.132668|

### Analisis Hasil
1. KNN (K-Nearest Neighbors):<br>
MSE Train: 0.066774
MSE Test: 0.087716
Analisis: KNN menunjukkan performa yang cukup baik dengan MSE yang relatif rendah pada data pelatihan dan pengujian. Namun, perbedaan antara MSE train dan test menunjukkan bahwa model ini mungkin mengalami sedikit overfitting, meskipun tidak signifikan.
2. Random Forest (RF): <br>
MSE Train: 0.001631
MSE Test: 0.01067
Analisis: Random Forest menunjukkan MSE yang sangat rendah pada data pelatihan, yang menunjukkan bahwa model ini dapat mempelajari pola dengan sangat baik. Namun, MSE pada data pengujian juga cukup rendah, menunjukkan bahwa model ini dapat generalisasi dengan baik dan tidak mengalami overfitting yang signifikan.
3. Boosting:<br>
MSE Train: 0.121198
MSE Test: 0.132668
Analisis: MSE untuk model Boosting lebih tinggi dibandingkan dengan KNN dan Random Forest, baik pada data pelatihan maupun pengujian. Ini menunjukkan bahwa meskipun Boosting adalah metode yang kuat, dalam konteks dataset ini, model ini tidak memberikan hasil yang optimal. 