# Keterbatasan dan Etika

## Keterbatasan metodologis

1. Evaluasi utama memakai satu chunk gold per QA. Jawaban klinis dapat tersebar pada beberapa chunk; analisis multi-gold hanya distant supervision dan belum divalidasi pakar.
2. Pertanyaan asli berasal dari delapan template. Parafrasa terkontrol meningkatkan variasi, tetapi tidak sepenuhnya mewakili bahasa pengguna nyata, typo, ambiguitas, atau pertanyaan majemuk.
3. Sepuluh split meningkatkan estimasi stabilitas internal, tetapi tidak menjadi validasi eksternal lintas rumah sakit, format laporan, atau populasi pasien.
4. Observasi pada seed dan skenario berbeda tidak sepenuhnya independen. Karena itu, hasil gabungan dilengkapi CI bootstrap terstratifikasi dan jumlah seed signifikan.
5. Dense retrieval dan reranker memakai model multilingual umum tanpa fine-tuning klinis. Kinerja model lain atau adaptasi domain dapat berbeda.
6. Dekomposisi komponen berurutan dan bergantung urutan; persentase kontribusi bukan efek kausal independen.
7. Audit kegagalan memakai proxy otomatis pada 20 QID dan tidak boleh dianggap adjudikasi klinis.
8. Generation tidak selesai karena browser computer-use tidak tersedia. Tidak ada bukti eksperimental bahwa retrieval yang lebih baik menghasilkan jawaban LLM yang lebih baik.

## Privasi dan tata kelola data

Laporan telah disamarkan, tetapi tetap diperlakukan sebagai data sensitif. Folder `data/`, `results/`, `models/`, `kaggle_io/`, kredensial, dan cache tidak dilacak Git. Tabel paper hanya berisi agregat. Kutipan singkat pada jejak manual dibatasi dan identitas eksplisit disamarkan.

Pemrosesan GPU yang pernah dilakukan menggunakan dataset Kaggle Private. Dataset sementara Tahap 3/6 dihapus setelah hasil lokal diverifikasi. Pada Tahap 11, kegagalan terjadi sebelum upload sehingga tidak ada data dikirim. Dataset klinis tidak boleh dibuat publik atau dimasukkan ke repository, supplementary material, issue tracker, maupun layanan lain tanpa persetujuan eksplisit dan dasar etik yang sesuai.

## Risiko penggunaan

Sistem adalah eksperimen information retrieval, bukan alat diagnosis atau rekomendasi diet. Rank tinggi tidak menjamin kutipan benar, lengkap, mutakhir, atau aman secara klinis. Pengguna klinis tetap harus memeriksa laporan sumber. Hasil penelitian tidak boleh digunakan untuk keputusan pasien tanpa validasi prospektif, pengawasan tenaga kesehatan, audit bias, dan mekanisme keselamatan.

## Bias dan representasi

Distribusi tipe tidak seimbang: pedoman jauh lebih sedikit daripada tujuh tipe lain. Data berasal dari satu koleksi dan mungkin mencerminkan praktik dokumentasi, populasi, istilah, serta kualitas pencatatan tertentu. Metrik agregat dapat menyembunyikan variasi antarpenyakit, usia, dan kompleksitas laporan; variabel tersebut tidak dianalisis untuk menghindari perluasan penggunaan data sensitif.

## Pelaporan yang bertanggung jawab

- Laporkan hasil negatif: P-QA merugikan NEW/PNEW dan alternatif dense tidak lolos kendala.
- Nyatakan ORACLE-W sebagai batas atas retrospektif, bukan metode praktis.
- Jangan mengisi atau memperkirakan metrik generation yang tidak tersedia.
- Jangan menyatakan multi-gold sebagai anotasi pakar.
- Verifikasi seluruh rujukan dari sumber asli sebelum publikasi.
- Laporkan approval etik, consent/waiver, dan kebijakan institusi sesuai dokumen resmi yang dimiliki peneliti; repository ini tidak menetapkannya.

