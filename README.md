# Laporan Proyek Machine Learning - Sistem Rekomendasi Film Netflix
Putu Yoga Suartana - MC298D5Y2265

## Project Overview
Seiring dengan pesatnya perkembangan teknologi digital, platform streaming film seperti Netflix, Disney+, dan lainnya telah menjadi sumber hiburan utama bagi jutaan orang di seluruh dunia.  Platform ini menyediakan katalog berisi ribuan hingga puluhan ribu judul film dan serial TV, yang ironisnya dapat menimbulkan masalah baru bagi pengguna: paradox of choice atau kebingungan dalam memilih tontonan.  Ketika dihadapkan pada terlalu banyak pilihan, pengguna sering kali kesulitan untuk memutuskan apa yang akan ditonton, yang dapat mengurangi kepuasan dan pengalaman mereka. 

Untuk mengatasi masalah ini, sistem rekomendasi menjadi komponen krusial yang tidak terpisahkan dari layanan streaming.  Sistem ini bertujuan untuk mempersonalisasi pengalaman pengguna dengan menyarankan konten yang paling relevan bagi setiap individu.  Pentingnya sistem rekomendasi ini telah terbukti secara bisnis. Sebagai contoh, penelitian yang dipublikasikan oleh eksekutif Netflix, Carlos A. Gomez-Uribe dan Neil Hunt, menyatakan bahwa lebih dari 80% jam tayang di Netflix didorong oleh sistem rekomendasi mereka.  Hal ini menunjukkan bahwa rekomendasi yang akurat dan relevan tidak hanya meningkatkan kepuasan pengguna tetapi juga secara langsung berkontribusi pada retensi pelanggan dan keberhasilan platform. 

Proyek akhir ini bertujuan untuk merancang dan membangun model sistem rekomendasi film menggunakan dataset publik.  Dengan menerapkan teknik machine learning yang telah dipelajari, proyek ini akan mengeksplorasi dua pendekatan utama dalam sistem rekomendasi untuk menghasilkan daftar tontonan yang dipersonalisasi.

**Referensi:**

Gomez-Uribe, C. A., & Hunt, N. (2016). The Netflix recommender system: Algorithms, business value, and innovation. ACM Transactions on Management Information Systems (TMIS), 6(4), 1-19. Dapat diakses melalui: [ACM Digital Library](https://dl.acm.org/doi/10.1145/2843948).

## Business Understanding
Bagian ini menguraikan proses klarifikasi masalah, tujuan, dan pendekatan solusi yang akan diimplementasikan untuk proyek ini. 

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan, masalah utama yang akan diselesaikan adalah:

* Bagaimana cara membangun sistem yang dapat merekomendasikan film-film dengan konten yang serupa (misalnya, genre, alur cerita, atau pemeran yang sama) dengan film yang disukai pengguna? 
* Bagaimana perbedaan teknik representasi fitur (vektorisasi teks) dapat memengaruhi kualitas dan karakteristik rekomendasi yang dihasilkan oleh model berbasis konten? 

### Goals
Tujuan dari proyek ini adalah untuk menjawab pernyataan masalah di atas dengan:

* Menciptakan sebuah model sistem rekomendasi yang mampu memberikan rekomendasi film berdasarkan kemiripan konten dari metadata yang tersedia. 
* Mengembangkan dan membandingkan dua model Content-Based Filtering yang menggunakan teknik vektorisasi berbeda (TF-IDF dan CountVectorizer) untuk memahami dampaknya terhadap hasil rekomendasi.

### Solution Approach
Untuk mencapai tujuan tersebut, proyek ini akan mengembangkan dan membandingkan dua model sistem rekomendasi dengan pendekatan Content-Based Filtering, namun menggunakan teknik representasi fitur yang berbeda: 

1. **Content-Based Filtering dengan TF-IDF:** Solusi pertama akan menganalisis metadata film dan menggunakan teknik TF-IDF untuk pembobotan fitur.  Kemiripan antar film akan dihitung menggunakan Cosine Similarity. 
2. **Content-Based Filtering dengan CountVectorizer:** Solusi kedua akan menggunakan pendekatan yang sama, namun dengan teknik vektorisasi CountVectorizer (Bag of Words) sebagai pembanding untuk menganalisis dampaknya terhadap hasil rekomendasi.

## Data Understanding
Pada tahap ini, akan dilakukan eksplorasi mendalam terhadap dataset Netflix Movies and TV Shows untuk memahami karakteristik, distribusi, dan potensi pola yang ada di dalamnya.

**Sumber Dataset:** Dataset ini diperoleh dari platform Kaggle dan dapat diakses melalui tautan berikut: [Netflix Movies and TV Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows).

Dataset ini terdiri dari satu file CSV yang berisi 8807 baris data dan 12 kolom. Dari inspeksi awal, diketahui bahwa beberapa kolom seperti `director`, `cast`, dan `country` memiliki nilai yang hilang (missing values). Fitur-fitur utama seperti `title`, `listed_in` (sebagai genre), dan 'description' memiliki data yang lengkap dan akan menjadi dasar utama dalam pembuatan sistem rekomendasi.

### Deskripsi Variabel
Berikut adalah deskripsi untuk setiap variabel yang relevan dalam proyek ini:

* `show_id`: Nomor identifikasi unik untuk setiap judul.
* `type`: Tipe konten (Movie atau TV Show).
* `title`: Judul konten.
* `director`: Nama sutradara.
* `cast`: Daftar pemeran utama.
* `country`: Negara tempat produksi.
* `date_added`: Tanggal konten ditambahkan ke Netflix.
* `release_year`: Tahun rilis konten.
* `rating`: Klasifikasi rating usia (misalnya, TV-MA, PG-13).
* `duration`: Durasi film (dalam menit) atau jumlah season (untuk TV Show).
* `listed_in`: Kategori atau genre konten.
* `description`: Sinopsis atau ringkasan singkat dari alur cerita.

### Exploratory Data Analysis (EDA)
**Distribusi Genre Konten**

**Insight:** Genre International Movies dan Dramas merupakan kategori yang paling dominan dalam dataset, diikuti oleh Comedies. Ini menunjukkan fokus Netflix pada konten internasional dan narasi drama.

**Distribusi Rating Konten**

**Insight:** Mayoritas konten di Netflix memiliki rating TV-MA (Mature Audience) dan TV-14 (Parents Strongly Cautioned), menandakan bahwa audiens utama platform ini adalah penonton dewasa dan remaja.

## Data Preparation
Pada tahap ini, dilakukan proses persiapan data untuk mengubah data mentah menjadi format yang bersih dan siap digunakan untuk pemodelan. Proses ini sangat penting untuk memastikan kualitas dan relevansi fitur yang akan diolah oleh model. Teknik-teknik yang diterapkan meliputi:

1. **Penanganan Nilai yang Hilang:** Integritas data dipastikan dengan mengisi nilai yang hilang pada kolom-kolom kunci seperti `director` dan `cast` dengan string kosong. Pendekatan ini mencegah error pada tahap pemrosesan teks tanpa mengurangi jumlah data.

2. **Rekayasa dan Ekstraksi Fitur:** Untuk proyek ini, data difokuskan hanya pada tipe 'Movie'. Fitur-fitur teks yang relevan seperti genre, deskripsi, pemeran, dan sutradara dipilih untuk digabungkan.

3. **Pembersihan dan Unifikasi Fitur:** Untuk memastikan setiap entitas (seperti nama orang atau genre) diperlakukan sebagai satu token unik, spasi di dalamnya dihilangkan (contoh: 'David Fincher' menjadi 'DavidFincher'). Selanjutnya, semua fitur teks yang relevan—deskripsi, genre, pemeran, dan sutradara—digabungkan menjadi satu kolom komprehensif yang disebut `tags`. Kolom `tags` ini berfungsi sebagai representasi konten holistik untuk setiap film, yang akan menjadi dasar utama bagi model Content-Based Filtering.

Setelah melalui seluruh tahapan di atas, dataset kini telah siap untuk dilanjutkan ke proses pemodelan.

## Modeling and Result
Pada tahap ini, dilakukan pengembangan model sistem rekomendasi dengan pendekatan Content-Based Filtering. Pendekatan ini dipilih karena dapat memberikan rekomendasi berdasarkan atribut internal dari item (film) itu sendiri, seperti genre, sinopsis, dan pemeran. Dua variasi model dikembangkan untuk mengeksplorasi pengaruh teknik representasi fitur yang berbeda terhadap hasil rekomendasi.

### Model 1: Content-Based Filtering dengan TF-IDF
Model pertama memanfaatkan teknik TF-IDF (Term Frequency-Inverse Document Frequency) untuk proses vektorisasi.

1. **Proses:** Kolom `tags` yang berisi gabungan semua fitur teks diubah menjadi matriks vektor numerik. TF-IDF memberikan bobot pada setiap kata berdasarkan frekuensinya dalam sebuah film dan keunikannya di seluruh koleksi film. Setelah itu, metrik Cosine Similarity digunakan untuk menghitung skor kemiripan antara setiap pasang vektor film.

2. **Tujuan:** Model ini bertujuan menghasilkan rekomendasi berdasarkan kemiripan konten yang sudah dibobotkan, di mana kata-kata yang lebih spesifik dan unik memiliki pengaruh lebih besar.

### Model 2: Content-Based Filtering dengan CountVectorizer
Sebagai alternatif, model kedua dikembangkan menggunakan teknik CountVectorizer (atau Bag of Words).

1. **Proses:** Sama seperti model pertama, kolom `tags` diubah menjadi matriks vektor. Namun, CountVectorizer hanya menghitung frekuensi kemunculan setiap kata tanpa mempertimbangkan bobot keunikannya. Matriks kemiripan kemudian dihitung menggunakan Cosine Similarity.

2. **Tujuan:** Model ini diimplementasikan sebagai pembanding untuk menganalisis bagaimana representasi fitur yang lebih sederhana (hanya berdasarkan frekuensi) memengaruhi hasil akhir rekomendasi.

### Hasil Rekomendasi
Kedua model yang telah dibangun mampu menghasilkan daftar top-5 rekomendasi film berdasarkan judul film yang diberikan sebagai input. Berikut adalah contoh hasil perbandingan untuk film 'The Social Network':

**Rekomendasi dari Model 1 (TF-IDF):**

* The Great Hack
* Steve Jobs
* The Hater
* The Circle
* Inside Job

**Rekomendasi dari Model 2 (CountVectorizer / Bag of Words):**

* The Great Hack
* Steve Jobs
* The Hater
* The Circle
* Inside Job

### Analisis Hasil:

Hasil yang identik dari kedua model mengindikasikan bahwa fitur-fitur yang telah direkayasa pada kolom tags (seperti nama pemeran dan sutradara) sangat kuat dan dominan, sehingga pilihan antara teknik vektorisasi TF-IDF dan CountVectorizer tidak menghasilkan perbedaan yang signifikan untuk kasus uji ini.

### Kelebihan dan Kekurangan Pendekatan
Pendekatan Content-Based Filtering memiliki beberapa kelebihan dan kekurangan sebagai berikut:

**Kelebihan:**

* Tidak Memerlukan Data Pengguna Lain: Efektif tanpa riwayat interaksi pengguna lain.

* Mampu Merekomendasikan Item Spesifik (Niche): Dapat merekomendasikan item yang tidak populer.

* Transparan dan Mudah Dijelaskan: Alasan di balik rekomendasi dapat dijelaskan dengan mudah.

**Kekurangan:**

* Terbatas pada Konten yang Sudah Ada: Cenderung merekomendasikan item yang sangat mirip dan kurang memberikan kejutan (serendipity).

* Ketergantungan pada Kualitas Fitur: Kualitas rekomendasi sangat bergantung pada kelengkapan metadata.

## Evaluation
Evaluasi model dilakukan secara kuantitatif menggunakan metrik Precision.

### Formula dan Cara Kerja
* **Formula:** `Precision = (Jumlah Film Rekomendasi yang Relevan) / (Jumlah Total Film yang Direkomendasikan)`

* **Definisi Relevan:** Sebuah film rekomendasi dianggap "relevan" jika memiliki setidaknya satu genre yang sama dengan film acuan.

### Hasil Evaluasi
Berdasarkan pengujian pada film 'The Social Network' (genre: Dramas), diperoleh hasil sebagai berikut:

* **Model 1 (TF-IDF):** Memperoleh nilai Precision @5 sebesar 1.00.

* **Model 2 (CountVectorizer):** Juga memperoleh nilai Precision @5 sebesar 1.00.

Nilai 1.00 ini berarti bahwa 5 dari 5 film yang direkomendasikan oleh masing-masing model terbukti relevan. Hasil ini secara kuantitatif membuktikan bahwa kedua model mampu memberikan rekomendasi dengan tingkat presisi yang sangat tinggi.

## Kesimpulan
Proyek ini telah berhasil mengembangkan sistem rekomendasi film menggunakan pendekatan Content-Based Filtering dengan dua teknik vektorisasi berbeda.  Kedua model menunjukkan kinerja yang kuat dengan nilai presisi 100% pada kasus uji, membuktikan efektivitasnya dalam merekomendasikan film berdasarkan kemiripan konten. Ditemukan juga bahwa rekayasa fitur yang cermat (pembuatan kolom `tags`) memiliki dampak yang sangat signifikan terhadap hasil. Dengan demikian, tujuan proyek untuk menciptakan dan membandingkan model rekomendasi berdasarkan konten telah tercapai dengan baik.
