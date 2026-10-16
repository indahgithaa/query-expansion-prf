# Draf Bab IV — Hasil dan Pembahasan

## 4.1 Hasil persiapan data

Audit awal menemukan 911 QA dari 120 laporan dan 13.737 chunk. Setelah 53 duplikat dihapus, diperoleh 858 QA. Semua QA berhasil dihubungkan ke satu chunk gold. Sebanyak 12 konteks tidak identik secara literal dengan teks chunk, tetapi identitas metadata tetap valid. Hanya 79 dari 911 `answer_start` mentah yang tepat sehingga offset tidak dipakai.

Split tingkat dokumen menghasilkan 72 dokumen koleksi, 24 dev, dan 24 test pada setiap seed. Tidak ada overlap dokumen dan delapan tipe pertanyaan muncul pada semua bagian. Sebanyak 770 QA memenuhi syarat multi-gold; pada ambang 0,5, 0,6, dan 0,7 diperoleh berturut-turut 4.905, 3.551, dan 2.115 qrels.

## 4.2 Pemilihan parameter dasar

Grid dev memilih `m=5` dan `r=3`. RM3 terbaik memakai lima dokumen feedback, 30 term, dan interpolasi 0,5. Pemilihan ini dilakukan sebelum evaluasi test. Pembobotan sigmoid kemudian disetel ulang ketika ukuran similarity atau retriever berubah dan pada setiap seed final.

## 4.3 Pemilihan komponen semantik

Pada dev seed 1–3, QS-BM25 mencapai MRR rata-rata 0,4031. QS-EMB dan QS-HYB-MM sedikit lebih tinggi, masing-masing 0,4080 dan 0,4082, tetapi keduanya menurunkan TPL dan NEW lebih dari 0,01. Dengan aturan ketahanan, QS-BM25 dipertahankan.

QS-EMB memiliki ketepatan tipe feedback PARA 0,7719, lebih tinggi daripada QS-BM25 0,5926. Namun ketepatan tipe saja tidak menjamin ranking akhir yang stabil. Isi feedback dan kecocokan dengan retriever target turut menentukan hasil.

Retriever R-BM25 mencapai MRR 0,4031. Dense teks penuh hanya 0,3063, sedangkan dense Rocchio terbaik 0,3358. Hybrid Rocchio terbaik mencapai 0,3986 dan memperbaiki NEW/PNEW, tetapi TPL turun 0,0626. Karena melewati batas penurunan, R-BM25 dipilih.

Reranker pada kandidat FINAL meningkatkan MRR dev dari 0,4031 menjadi 0,4171 pada lambda 0,25 tanpa menurunkan skenario. Lambda 0,50 memberi rata-rata 0,4217 tetapi menurunkan TPL 0,0345, sehingga ditolak. Konfigurasi FINAL dikunci sebelum test.

## 4.4 Hasil retrieval utama

| Metode | TPL | NEW | PARA | PNEW | Rata-rata |
|---|---:|---:|---:|---:|---:|
| B1 | 0,3085 | 0,3085 | 0,1871 | 0,1871 | 0,2478 |
| RM3 | 0,3171 | 0,3171 | 0,1931 | 0,1931 | 0,2551 |
| P-QA | 0,6137 | 0,1798 | 0,4016 | 0,1629 | 0,3395 |
| P-FUS | 0,5204 | 0,3062 | 0,3437 | 0,2551 | 0,3564 |
| ZONE3 | 0,6156 | 0,3126 | 0,3013 | 0,2411 | 0,3677 |
| SOFT | 0,6185 | 0,3469 | 0,3466 | 0,2365 | 0,3871 |
| B1+RR | 0,3376 | 0,3376 | 0,2943 | 0,2943 | 0,3160 |
| FINAL | 0,6178 | 0,3591 | 0,3584 | 0,2491 | **0,3961** |

FINAL menjadi metode non-oracle terbaik. Recall@10 mencapai 0,7206 dan nDCG@10 0,4650. B1 hanya mencapai Recall@10 0,5084 dan nDCG@10 0,2935. RM3 memberi peningkatan kecil, tetapi masih jauh di bawah SOFT dan FINAL.

Pada TPL, SOFT sedikit lebih tinggi daripada FINAL (0,6185 berbanding 0,6178). Pada tiga skenario lain, reranker menaikkan hasil sekitar 0,012. Ini sesuai prosedur seleksi: reranker boleh dinonaktifkan pada seed ketika tidak lolos kendala dev. Lambda FINAL bernilai 0,25 pada enam seed dan 0 pada empat seed.

## 4.5 Jawaban rumusan masalah pertama

PRF klasik RM3 hanya menaikkan MRR rata-rata 0,0073 dari B1. Pada PARA/PNEW, peningkatannya 0,0060. Hal ini menunjukkan bahwa feedback dari hasil awal $D_p$ tidak cukup informatif ketika query generik atau diparafrasakan. Hasil ini mendukung motivasi pemakaian sumber feedback lintas pasien melalui pertanyaan serupa.

## 4.6 Jawaban rumusan masalah kedua

SOFT meningkatkan B1 sebesar 0,1393 MRR, sedangkan FINAL meningkatkan B1 sebesar 0,1483. CI bootstrap FINAL–B1 adalah [0,1405; 0,1559]. Uji Wilcoxon signifikan setelah Holm pada 10/10 seed, dengan matched-pairs rank-biserial 0,5660.

FINAL juga mengungguli B1+RR sebesar 0,0801, CI [0,0719; 0,0880], serta RM3 sebesar 0,1410, CI [0,1331; 0,1488]. Karena pembanding B1+RR telah menerima model reranker yang sama, selisih 0,0801 mengisolasi bukti bahwa ekspansi adaptif memberi manfaat di luar reranking.

Dekomposisi berurutan mencatat +0,0917 dari QA feedback, +0,0169 dari feedback eksternal/fusion, +0,0308 dari sigmoid, dan +0,0089 dari reranker. Total ekspansi dan pembobotan adalah 93,97% kenaikan; reranker 6,03%. Persentase ini bergantung pada urutan dan tidak ditafsirkan sebagai efek kausal independen.

## 4.7 Jawaban rumusan masalah ketiga

FINAL mempunyai RI pooled 0,3791 dengan 4.268 win, 963 tie, dan 1.657 loss terhadap B1. RI per skenario selalu positif: 0,6045 pada TPL, 0,1870 pada NEW, 0,4646 pada PARA, dan 0,2602 pada PNEW.

P-QA memperlihatkan trade-off penting. Metode ini sangat kuat pada TPL, tetapi MRR NEW dan PNEW masing-masing hanya 0,1798 dan 0,1629; RI pada kedua skenario negatif. SOFT menahan risiko tersebut dengan menurunkan bobot QA ketika similarity rendah. Median bobot QA adalah 0,9828 pada TPL, 0,4035 pada NEW, 0,3843 pada PARA, dan 0,3028 pada PNEW.

Gap SOFT terhadap ORACLE-W rata-rata 0,0788. Gap terbesar terdapat pada PARA (0,1373), diikuti PNEW (0,0719), NEW (0,0678), dan TPL (0,0381). Model pembobotan sudah efektif pada template, tetapi belum selalu memilih sumber terbaik pada parafrasa.

## 4.8 Sensitivitas multi-gold

FINAL tetap menjadi metode non-oracle terbaik pada tiga ambang. MAP@50 FINAL turun secara wajar dari 0,4708 pada ambang 0,5 menjadi 0,4545 pada 0,7; nDCG@10 turun dari 0,5486 menjadi 0,5248. Korelasi Spearman urutan metode dengan MRR strict berada pada 0,9882–0,9941. Kesimpulan utama tidak bergantung pada satu ambang multi-gold.

## 4.9 Kinerja menurut tipe pertanyaan

FINAL mengungguli B1 pada seluruh tipe. MRR FINAL per tipe adalah 0,3790 untuk skrining, 0,3873 status gizi, 0,2593 biokimia, 0,4354 PES, 0,5014 tujuan diet, 0,4375 prinsip diet, 0,3805 syarat diet, dan 0,3703 pedoman.

PES memperoleh kenaikan terbesar, +0,3027. Pertanyaan diagnosis PES sangat diuntungkan oleh jawaban dari pertanyaan serupa yang mengandung struktur problem–etiology–sign/symptom. Biokimia tetap terendah karena bukti laboratorium dapat tersebar dan bentuk angka/satuan bervariasi. Pedoman memiliki RI tinggi tetapi sampelnya paling sedikit, sehingga simpangan antarseed lebih besar.

## 4.10 Analisis kesalahan

Gold FINAL berada di luar top-10 pada 1.926 dari 6.888 observasi (27,96%). Sampel deterministik 20 QID menunjukkan 11 kategori utama jawaban tersebar, dua feedback salah tipe, satu parafrasa tidak dikenali, dan enam lainnya. Dalam indikator yang boleh bertumpang tindih, 13 kasus mempunyai lebih dari satu chunk multi-gold dan sepuluh mempunyai bobot QA rendah.

Hasil tersebut mengingatkan bahwa label satu chunk dapat meremehkan retrieval yang menemukan bukti pendukung di chunk lain. Namun multi-gold otomatis juga dapat menghasilkan false positive. Audit pakar diperlukan untuk membedakan kegagalan retrieval, masalah chunking, dan ketidaklengkapan label.

## 4.11 Jejak perhitungan manual

Kasus seed 7, Q0100, TPL dipilih karena dokumennya hanya mempunyai 32 chunk dan kenaikan B1–SOFT terbesar dari 20 kandidat yang memenuhi syarat. Query setelah stopword menjadi `[menyusun, diagnosis, gizi, format, pes]`. QS-BM25 menghasilkan similarity maksimum 9,212502. Dengan $k=1$ dan $\tau=3{,}577024$, bobot QA adalah 0,996444.

Chunk gold mempunyai skor normalisasi QA 1 dan eksternal 0,926731, sehingga skor SOFT-nya

$$0{,}996444(1)+0{,}003556(0{,}926731)=0{,}999739.$$

Gold naik dari rank 14 pada B1 ke rank 1 pada SOFT dan tetap rank 1 pada FINAL. RR naik dari 0,071429 ke 1 dan nDCG@10 dari 0 ke 1. Jejak term BM25, normalisasi, tiga chunk, serta skor reranker lengkap tersedia di `reports/12_manual_trace.md`.

## 4.12 Evaluasi generation

Generation tidak menghasilkan output karena kontrol browser Kaggle tidak tersedia melalui tiga jalur yang dicoba. Tahap ditandai SKIPPED sesuai protokol opsional. Input dan notebook tetap tersedia secara privat, tetapi tidak ada nilai ROUGE-L, BERTScore, abstain rate, atau Wilcoxon generation. Oleh karena itu, pembahasan dibatasi pada retrieval.

## 4.13 Keterbatasan pembahasan

Generalisasi dibatasi satu kumpulan laporan dan pertanyaan templatik. Parafrasa bersifat terkontrol. Multi-gold belum divalidasi pakar. Split berulang mengukur stabilitas pembagian internal, bukan transfer lintas institusi. Reranker dan embedding memakai model umum multilingual tanpa adaptasi domain. Hasil tidak boleh dipakai sebagai rekomendasi klinis atau bukti keselamatan sistem.

## 4.14 Implikasi

Secara teknis, hasil mendukung penggunaan dua sumber feedback dengan pengendali berbasis kemiripan, bukan pemakaian QA feedback secara selalu aktif. Secara praktis, pipeline memperbesar probabilitas chunk gold masuk top-10 dari 50,84% menjadi 72,06%. Sistem masih memerlukan validasi manusia, terutama pada PNEW, biokimia, dan jawaban yang tersebar.

Seluruh angka bab ini berasal dari artefak `results/` dan dapat dibangun ulang dengan script repository. Detail rujukan teori dan interpretasi domain perlu diverifikasi pengguna sebelum naskah final.

