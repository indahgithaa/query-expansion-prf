# Pemilihan Komponen Semantik di Dev

## Protokol pemilihan

Pemilihan hanya menggunakan partisi dev pada seed 1–3. Empat skenario (TPL, NEW, PARA, dan PNEW) menghasilkan 2.052 observasi query–skenario. Data test seed 1–10 hanya ikut dicakup saat menyiapkan cache reranker agar Tahap 7 tidak perlu menjalankan model kembali; skor atau label test tidak pernah dibaca untuk memilih komponen maupun parameter.

Setiap langkah memaksimalkan rata-rata MRR gold ketat pada empat skenario, dengan syarat MRR tiap skenario tidak boleh turun lebih dari 0,01 terhadap pilihan langkah sebelumnya. Parameter sigmoid `k` dan `tau` disetel ulang pada dev untuk setiap kandidat ukuran question similarity dan retriever.

## Cache semantik

Embedding teks penuh dibuat dengan `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a`. Sebanyak 2.205 teks unik di-encode menjadi vektor 768 dimensi yang ternormalisasi pada Tesla T4 dalam 34,58 detik, lalu diekspansi menjadi 4.104 pemakaian query.

Cache reranker menggunakan `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; model cadangan tidak dipakai. Input mencakup top-20 B1 dan seluruh kandidat pipeline yang dibutuhkan pada dev/test seed 1–10 di semua skenario. Setelah deduplikasi terdapat 61.011 pasangan unik dari 482.743 kemunculan kandidat. Scoring selesai dalam 933,06 detik pada Tesla T4.

- SHA-256 input: `a505a11017501d96b244eb3093bceccd6cdab6c22b46a53a7cac34ebc8cb413c`
- SHA-256 skor: `25b72753dc394ec6c2eb911216a03f9ab085dee87aa8ef8375fac125e16a78a9`
- Pertanyaan yang diberikan ke reranker selalu pertanyaan asli; teks kedua adalah teks chunk kandidat.

## Langkah 2: ukuran question similarity

| Kandidat | k | tau | TPL | NEW | PARA | PNEW | Rata-rata | Memenuhi kendala |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| QS-BM25 | 2 | 3,1251 | 0,6133 | 0,3587 | 0,3577 | 0,2826 | 0,4031 | Ya |
| QS-EMB | 8 | 0,9451 | 0,5793 | 0,3225 | 0,4449 | 0,2853 | 0,4080 | Tidak |
| QS-HYB-MM | 8 | 0,9485 | 0,5775 | 0,3173 | 0,4676 | 0,2703 | 0,4082 | Tidak |
| QS-HYB-RRF | 8 | 0,0226 | 0,5577 | 0,3028 | 0,4603 | 0,2583 | 0,3948 | Tidak |

QS-EMB dan QS-HYB-MM memberi rata-rata sedikit lebih tinggi, tetapi keduanya menurunkan TPL dan NEW lebih dari 0,01; QS-HYB-MM juga menurunkan PNEW lebih dari 0,01. Karena hanya QS-BM25 yang memenuhi kendala, QS-BM25 dipertahankan.

## Langkah 3: retriever chunk

| Kandidat ringkas | k | tau | TPL | NEW | PARA | PNEW | Rata-rata | Memenuhi kendala |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| R-BM25 | 2 | 3,1251 | 0,6133 | 0,3587 | 0,3577 | 0,2826 | 0,4031 | Ya |
| R-DENSE teks penuh | 1 | 3,5972 | 0,4436 | 0,2510 | 0,2907 | 0,2398 | 0,3063 | Tidak |
| R-HYB teks penuh | 8 | 3,1251 | 0,5599 | 0,3204 | 0,3350 | 0,2665 | 0,3704 | Tidak |
| R-DENSE Rocchio terbaik | 2 | 8,4090 | 0,4365 | 0,3398 | 0,2838 | 0,2832 | 0,3358 | Tidak |
| R-HYB Rocchio terbaik secara rata-rata (`alpha=0,5; beta=0,25`) | 2 | 3,1251 | 0,5508 | 0,3774 | 0,3575 | 0,3087 | 0,3986 | Tidak |

Semua varian dense dan hybrid melanggar batas penurunan, terutama pada TPL. Karena R-BM25 menjadi satu-satunya kandidat yang memenuhi kendala, retriever tetap R-BM25. Tabel lengkap seluruh kombinasi Rocchio tersimpan secara privat di `results/stage6/retriever_candidates.csv`.

## Langkah 4: reranker

### B1 + reranker

| lambda | TPL | NEW | PARA | PNEW | Rata-rata | Memenuhi kendala |
|---:|---:|---:|---:|---:|---:|:---:|
| 0,00 | 0,2994 | 0,2994 | 0,1890 | 0,1890 | 0,2442 | Ya |
| 0,25 | 0,3185 | 0,3185 | 0,2146 | 0,2146 | 0,2666 | Ya |
| 0,50 | 0,3300 | 0,3300 | 0,2555 | 0,2555 | 0,2928 | Ya |
| **0,75** | **0,3422** | **0,3422** | **0,2944** | **0,2944** | **0,3183** | **Ya** |
| 1,00 | 0,3276 | 0,3276 | 0,3025 | 0,3025 | 0,3151 | Ya |

Untuk baseline B1+RR dipilih `lambda=0,75`, karena memberi rata-rata tertinggi dan tidak menurunkan skenario mana pun terhadap B1 tanpa reranker.

### Kandidat FINAL + reranker

| lambda | TPL | NEW | PARA | PNEW | Rata-rata | Memenuhi kendala |
|---:|---:|---:|---:|---:|---:|:---:|
| 0,00 | 0,6133 | 0,3587 | 0,3577 | 0,2826 | 0,4031 | Ya |
| **0,25** | **0,6155** | **0,3698** | **0,3792** | **0,3037** | **0,4171** | **Ya** |
| 0,50 | 0,5788 | 0,3811 | 0,3965 | 0,3303 | 0,4217 | Tidak |
| 0,75 | 0,5105 | 0,3774 | 0,3906 | 0,3489 | 0,4068 | Tidak |
| 1,00 | 0,3822 | 0,3386 | 0,3448 | 0,3275 | 0,3483 | Tidak |

Walaupun `lambda=0,50` memberi rata-rata tertinggi tanpa kendala, TPL turun 0,0345 dari pilihan sebelumnya. Oleh sebab itu, `lambda=0,25` dipilih: seluruh skenario meningkat dan rata-rata naik dari 0,4031 menjadi 0,4171.

## Konfigurasi FINAL yang dikunci

- `m=5`, `r=3`
- Question similarity: QS-BM25
- Retriever chunk: R-BM25
- Sigmoid: `k=2`, `tau=3,125116827123538`
- Reranker: `BAAI/bge-reranker-v2-m3`, top-20, `lambda=0,25`
- B1+RR untuk pembanding: top-20, `lambda=0,75`

Konfigurasi ini tersimpan di `configs/final.yaml` dengan `locked_before_stage7: true` dan `test_used_for_selection: false`. Konfigurasi tidak boleh diubah setelah hasil test dilihat; penyetelan ulang per pengulangan pada Tahap 7 tetap wajib memakai dev saja sesuai protokol.

## Verifikasi

- Impor cache reranker: `PASS`, 61.011 skor, revision dan kedua hash cocok.
- Seluruh pasangan yang diminta tersedia di cache; tidak ada fallback model.
- Grid pemilihan mencatat `selection_partition=dev`, seed 1–3, dan `test_used_for_selection=false`.
- Seluruh test lokal lulus setelah implementasi komponen semantik dan reranker.
- Sesi GPU dihentikan dan kedua dataset privat sementara Kaggle dihapus setelah artefak lokal terverifikasi.
