# Laporan Akhir Penelitian

## Ringkasan eksekutif

Penelitian ini mengevaluasi query expansion berbasis pseudo-relevance feedback (PRF) dengan question similarity untuk retrieval laporan asuhan gizi klinis. Evaluasi dilakukan pada 858 pasangan tanya-jawab dari 120 laporan, dengan split tingkat dokumen 60/20/20 yang diulang sepuluh kali. Empat skenario menguji pertanyaan template (TPL), topik yang tidak tersedia dalam koleksi feedback (NEW), parafrasa (PARA), dan gabungan parafrasa dengan topik baru (PNEW). Seluruh keputusan metode dan parameter dibuat pada data dev sebelum hasil test dilihat.

Konfigurasi FINAL memakai lima jawaban dan lima chunk eksternal sebagai feedback (`m=5`), pengulangan query tiga kali (`r=3`), QS-BM25, R-BM25, bobot sigmoid yang disetel per seed pada dev, serta reranker `BAAI/bge-reranker-v2-m3` pada top-20 bila lolos kendala dev. FINAL mencapai MRR 0,3961 ± 0,0232, Recall@10 0,7206 ± 0,0404, dan nDCG@10 0,4650 ± 0,0280. Baseline utama B1 mencapai MRR 0,2478, sedangkan RM3 mencapai 0,2551.

Kenaikan FINAL terhadap B1 adalah 0,1483 MRR dengan bootstrap CI 95% [0,1405; 0,1559], signifikan setelah koreksi Holm pada analisis gabungan dan pada 10/10 seed. FINAL juga lebih tinggi 0,1410 dari RM3 dan 0,0801 dari B1 yang diberi reranker. Dari dekomposisi berurutan, ekspansi serta pembobotan adaptif menyumbang 93,97% kenaikan B1–FINAL; reranker menyumbang 6,03%. Tahap generation bersifat opsional dan ditandai SKIPPED karena tiga jalur browser computer-use tidak tersedia sebelum data diunggah. Karena itu, penelitian ini menyimpulkan dampak pada kualitas retrieval, bukan pada kualitas jawaban LLM.

## Jawaban rumusan masalah

### 1. Efektivitas PRF klasik

RM3 hanya memberi kenaikan kecil terhadap B1. MRR rata-rata naik dari 0,2478 menjadi 0,2551 (+0,0073). Peningkatan terutama tampak pada TPL/NEW, dari 0,3085 menjadi 0,3171, sedangkan PARA/PNEW hanya naik dari 0,1871 menjadi 0,1931. Recall@10 rata-rata RM3 adalah 0,5472 dibanding 0,5084 untuk B1.

Secara statistik, FINAL–RM3 menghasilkan delta 0,1410 dengan CI 95% [0,1331; 0,1488], sehingga PRF klasik jauh tertinggal dari metode yang memanfaatkan feedback lintas pasien. Hasil ini konsisten dengan masalah awal: ketika query pendek dan generik, hasil awal di laporan pasien sendiri tidak selalu menjadi sumber feedback yang informatif. RM3 tetap sedikit lebih baik daripada B1, tetapi tidak cukup untuk mengatasi variasi parafrasa.

### 2. Peningkatan oleh PRF dengan question similarity

SOFT tanpa reranker memperoleh MRR 0,3871, atau +0,1393 terhadap B1. CI 95% delta adalah [0,1317; 0,1471], p Holm 1,25×10^-264, dan matched-pairs rank-biserial 0,5203. FINAL meningkatkan MRR menjadi 0,3961, atau +0,1483 terhadap B1. Efek FINAL–B1 memiliki rank-biserial 0,5660 dan signifikan pada 10/10 seed.

Perbandingan dengan B1+RR penting untuk memisahkan manfaat reranker dari query expansion. B1+RR mencapai MRR 0,3160, masih 0,0801 di bawah FINAL; CI 95% selisih [0,0719; 0,0880]. Dengan demikian, keunggulan FINAL tidak dapat dijelaskan hanya oleh reranker. Dekomposisi berurutan menunjukkan QA feedback +0,0917, penambahan feedback eksternal dan fusion +0,0169, pembobotan sigmoid +0,0308, dan reranker +0,0089. Tiga komponen ekspansi/pembobotan menyumbang 93,97% total kenaikan, sedangkan reranker 6,03%.

Hasil multi-gold mendukung kesimpulan yang sama. Pada ambang 0,5, FINAL memperoleh MAP@50 0,4708 dan nDCG@10 0,5486; pada ambang 0,6 nilainya 0,4606 dan 0,5370; pada ambang 0,7 nilainya 0,4545 dan 0,5248. Korelasi urutan metode terhadap MRR strict berada pada 0,9882–0,9941. FINAL selalu menjadi metode non-oracle terbaik.

### 3. Ketahanan dan pengaruh komponen

FINAL memperbaiki B1 pada seluruh skenario. MRR FINAL adalah 0,6178 pada TPL, 0,3591 pada NEW, 0,3584 pada PARA, dan 0,2491 pada PNEW. Dibanding B1, kenaikan masing-masing adalah +0,3093, +0,0506, +0,1713, dan +0,0620. Skenario PNEW tetap paling sulit.

Robustness Index pooled FINAL terhadap B1 adalah 0,3791 dari 4.268 win, 963 tie, dan 1.657 loss. Nilainya positif pada TPL (0,6045), NEW (0,1870), PARA (0,4646), dan PNEW (0,2602). Sebaliknya, P-QA tanpa adaptasi memiliki RI keseluruhan -0,0157 karena kuat pada TPL tetapi sering merugikan NEW dan PNEW. Temuan ini menunjukkan fungsi utama sigmoid: mengurangi ketergantungan pada QA feedback ketika question similarity rendah.

QS-EMB dan hybrid meningkatkan PARA, tetapi melanggar batas maksimum penurunan 0,01 pada TPL/NEW. QS-BM25 karena itu tetap dipilih. Retriever dense murni tertinggal dari BM25; hybrid Rocchio mendekati rata-rata BM25 tetapi menurunkan TPL 0,0626. R-BM25 tetap dipilih. Reranker meningkatkan B1 sebesar 0,0682 MRR, tetapi hanya menambah 0,0089 setelah SOFT, yang menunjukkan bahwa sebagian besar informasi relevan sudah dipindahkan ke atas oleh ekspansi adaptif.

Per tipe pertanyaan, FINAL mengungguli B1 pada seluruh delapan tipe. MRR tertinggi terjadi pada `tujuan_diet` (0,5014); terendah pada `biokimia` (0,2593). Kenaikan terbesar terdapat pada `pes` (+0,3027). Tipe `pedoman` mempunyai RI tinggi tetapi hanya 188 observasi sehingga ketidakpastiannya lebih besar.

## Konfigurasi FINAL

| Komponen | Pilihan |
|---|---|
| Feedback | 5 jawaban QA + 5 chunk eksternal |
| Pengulangan query | `r=3` |
| Question similarity | QS-BM25 pada pertanyaan tanpa penghapusan stopword |
| Retriever chunk | R-BM25, `k1=1,5`, `b=0,75` |
| Pembobotan | Sigmoid; `k` dan `tau` dipilih ulang pada dev setiap seed |
| Reranker | `BAAI/bge-reranker-v2-m3`, top-20 |
| Lambda FINAL | Dipilih pada dev; 0,25 pada 6 seed dan 0 pada 4 seed |
| Tie-breaking | Stabil untuk chunk; acak deterministik untuk question similarity |

QS-BM25 dan R-BM25 dipilih bukan karena semua alternatif memiliki rata-rata lebih rendah, melainkan karena hanya keduanya yang memenuhi kendala ketahanan antarskenario. Konfigurasi dikunci di `configs/final.yaml` sebelum run test.

## Hasil utama

| Metode | TPL | NEW | PARA | PNEW | MRR rata-rata | Recall@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B1 | 0,3085 | 0,3085 | 0,1871 | 0,1871 | 0,2478 | 0,5084 | 0,2935 |
| RM3 | 0,3171 | 0,3171 | 0,1931 | 0,1931 | 0,2551 | 0,5472 | 0,3088 |
| B1+RR | 0,3376 | 0,3376 | 0,2943 | 0,2943 | 0,3160 | 0,6118 | 0,3758 |
| SOFT | 0,6185 | 0,3469 | 0,3466 | 0,2365 | 0,3871 | 0,7052 | 0,4537 |
| FINAL | 0,6178 | 0,3591 | 0,3584 | 0,2491 | **0,3961** | **0,7206** | **0,4650** |
| ORACLE-W | 0,6567 | 0,4148 | 0,4839 | 0,3084 | 0,4659 | 0,7760 | 0,5326 |

ORACLE-W adalah batas atas retrospektif dan bukan metode deployable. Jarak FINAL ke ORACLE-W masih 0,0699 MRR. Gap SOFT–ORACLE terbesar pada PARA (0,1373), sehingga pemilihan sumber feedback per query masih menjadi ruang perbaikan utama.

## Analisis kesalahan

Dari 6.888 observasi, 1.926 (27,96%) menempatkan gold FINAL di luar top-10. Audit 20 QID unik menemukan 11 kasus dengan kategori utama jawaban tersebar, dua feedback salah tipe, satu parafrasa tidak dikenali, dan enam kategori lainnya. Indikator non-eksklusif menemukan 13 kasus dengan lebih dari satu chunk multi-gold dan sepuluh parafrasa berbobot QA rendah. Temuan ini berbasis proxy otomatis, bukan penilaian klinis manual.

Jejak manual pada seed 7, Q0100, TPL memperlihatkan mekanisme konkret. Pada dokumen 32 chunk, gold berpindah dari rank 14 pada B1 ke rank 1 pada SOFT dan tetap rank 1 pada FINAL. RR berubah 0,071429 menjadi 1,000000, sedangkan nDCG@10 berubah dari 0 menjadi 1. Perhitungan lengkap tersedia di `reports/12_manual_trace.md`.

## Generation

Tahap 11 ditandai **SKIPPED**. Input privat 1.730 baris dan notebook Qwen2.5 4-bit telah disiapkan, tetapi Chrome, Edge, dan in-app browser sama-sama tidak tersedia bagi computer-use. Kegagalan terjadi sebelum login atau upload, sehingga tidak ada teks klinis yang meninggalkan repository. ROUGE-L, BERTScore, abstain rate, dan uji Wilcoxon generation tidak dilaporkan. Oleh sebab itu, penelitian tidak mengklaim bahwa kenaikan retrieval otomatis menaikkan kualitas jawaban generatif.

## Keterbatasan

1. Gold utama hanya satu chunk per QA, sedangkan jawaban dapat tersebar; multi-gold dibentuk melalui distant supervision dan belum divalidasi manual.
2. Delapan bentuk pertanyaan asli sangat templatik. Parafrasa dibuat terkontrol dan belum mewakili seluruh variasi pertanyaan pengguna nyata.
3. Pengulangan split berbagi sebagian QA antar-seed. Uji gabungan karena itu dilengkapi hasil per-seed dan bootstrap terstratifikasi, tetapi bukan pengganti evaluasi pada korpus eksternal independen.
4. Pemilihan model semantik dibatasi model gratis yang dapat dijalankan pada sumber daya tersedia. Dense retrieval tidak dituning dengan fine-tuning domain.
5. Audit kesalahan memakai indikator otomatis dan sampel 20 QID; interpretasi klinis memerlukan pakar dietisien.
6. Tahap generation tidak selesai. Dampak retrieval terhadap jawaban RAG belum terbukti.
7. Data klinis telah disamarkan tetapi tetap sensitif; artefak privat tidak boleh dipublikasikan.

## Saran

- Validasi manual subset multi-gold dan kategori error bersama dietisien.
- Uji classifier atau calibrator pemilih sumber feedback yang dilatih hanya pada dev untuk memperkecil gap terhadap ORACLE-W, khususnya PARA.
- Evaluasi parafrasa alami yang dikumpulkan dari pengguna dan korpus eksternal lintas institusi.
- Fine-tune retriever multilingual pada data domain tanpa mengubah test, lalu ulangi seluruh seleksi dari dev.
- Jalankan notebook generation ketika akses Kaggle tersedia dan tambahkan penilaian faktualitas serta evaluasi manusia, bukan hanya ROUGE/BERTScore.

## Tahap yang dilewati

Hanya Tahap 11 (generation di Kaggle) yang dilewati, dengan status SKIPPED sesuai aturan tahap opsional. Tahap 0–10 dan 12 selesai; Tahap 13 menyatukan seluruh hasil dan artefak reproduksi.

## Integritas dan rujukan

Data test tidak dipakai untuk memilih konfigurasi. Seluruh pemilihan komponen dilakukan pada dev seed 1–3; parameter per-seed dipilih pada dev sebelum fitur test dibangun. Konfigurasi final tidak diubah setelah hasil test dilihat.

Daftar rujukan konseptual dalam draf metodologi berasal dari rancangan penelitian. **Detail bibliografis, judul, tahun, DOI, nomor halaman, dan kesesuaian setiap sitasi harus diverifikasi pengguna dari sumber asli sebelum naskah diajukan.** Tidak ada klaim klinis baru yang diturunkan dari sistem retrieval ini.

## Artefak utama

- `reports/08_results.md`: tabel evaluasi dan statistik lengkap.
- `reports/09_components.md`: analisis komponen dan dekomposisi.
- `reports/10_analysis.md`: tipe pertanyaan, bobot, oracle gap, dan error.
- `reports/12_manual_trace.md`: jejak perhitungan manual.
- `reports/DRAFT_BAB3_METODOLOGI.md` dan `reports/DRAFT_BAB4_HASIL.md`: draf skripsi.
- `paper/`: outline, tabel, gambar, keterbatasan, etika, dan reproduksibilitas.

