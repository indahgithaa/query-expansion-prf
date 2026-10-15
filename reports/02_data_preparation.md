# Tahap 2 — Pra-pemrosesan, Split, dan Label Relevansi

Tanggal: 15 Oktober 2026

## Protokol

Pipeline `scripts/prepare_data.py` menerapkan normalisasi NFKC, mengubah pola `#Uxxxx` menjadi karakter Unicode, menghapus `U+2060`, lalu melakukan deduplikasi dengan kunci `(nama file ternormalisasi, question)`. Jika satu kunci memiliki beberapa baris, baris dengan ID numerik terkecil dipertahankan secara deterministik.

Artefak yang memuat teks klinis disimpan hanya di `data/processed/` dan diabaikan Git. Audit agregat mesin-baca berada di `results/data_preparation/audit.json`. Laporan ini tidak memuat nama laporan atau kutipan klinis.

## Rekonsiliasi

| Langkah | Jumlah QA |
|---|---:|
| Mentah | 911 |
| Setelah deduplikasi | 858 |
| Setelah join | 858 |
| Final | 858 |

Sebanyak 53 baris dibuang dari 35 kelompok duplikat. Dua puluh delapan kelompok duplikat menunjuk lebih dari satu lokasi gold; kebijakan ID terkecil dipakai tanpa meninjau test atau hasil retrieval. Semua 911 baris mentah dapat dihubungkan ke chunk sebelum deduplikasi dan seluruh 858 QA final memiliki tepat satu `gold_chunk_id`.

Satu nama file QA dan satu nama file chunk berubah setelah penghapusan `U+2060`; tidak ada collision. Dua belas QA final memiliki `context` sumber yang tidak identik dengan teks chunk. Audit menyimpan panjang dan hash kedua teks, sedangkan label gold tetap mengikuti identitas metadata.

## Data olahan

| Artefak privat | Isi | Jumlah |
|---|---|---:|
| `documents.jsonl` | Pemetaan dokumen | 120 |
| `chunks.jsonl` | Korpus chunk | 13.737 |
| `qa.jsonl` | QA deduplicated dan tipe | 858 |
| `qrels/strict.jsonl` | Satu gold ketat per QA | 858 |
| `splits/seed_01.json`–`seed_10.json` | Split dokumen dan QA | 10 |

`data/processed/manifest.json` menyimpan ukuran serta SHA-256 setiap artefak utama. Rerun menghasilkan manifest identik.

## Split dokumen

Setiap pengulangan menggunakan 72 dokumen koleksi, 24 dev, dan 24 test. Pengacakan deterministik dimulai dari seed 1–10 dan mengulang pengacakan hanya bila salah satu bagian tidak memuat kedelapan tipe. Semua seed valid pada percobaan pertama.

| Seed | QA koleksi | QA dev | QA test | Overlap dokumen | Semua tipe di tiap bagian |
|---:|---:|---:|---:|---:|:---:|
| 1 | 511 | 174 | 173 | 0 | Ya |
| 2 | 513 | 171 | 174 | 0 | Ya |
| 3 | 513 | 168 | 177 | 0 | Ya |
| 4 | 514 | 172 | 172 | 0 | Ya |
| 5 | 515 | 170 | 173 | 0 | Ya |
| 6 | 517 | 172 | 169 | 0 | Ya |
| 7 | 515 | 173 | 170 | 0 | Ya |
| 8 | 515 | 173 | 170 | 0 | Ya |
| 9 | 518 | 170 | 170 | 0 | Ya |
| 10 | 513 | 171 | 174 | 0 | Ya |

Pembagian dilakukan sepenuhnya pada tingkat dokumen: satu pasien tidak pernah muncul di lebih dari satu bagian dalam pengulangan yang sama.

## Label gold ketat

Label utama terdiri atas tepat satu chunk per QA, yaitu chunk yang ditunjuk oleh `(file, page_number, chunk_index)` setelah nama file dinormalisasi. Diperoleh 858 qrels untuk 858 QA; `answer_start` tidak digunakan.

## Label multi-gold

Token jawaban dibentuk dengan *lowercase*, regex `\w+`, daftar stopword Sastrawi, serta stopword tambahan `apa`, `bagaimana`, `yang`, `ini`, dan `pasien`. Himpunan token unik jawaban disebut `A`. Chunk dalam dokumen yang sama ditandai relevan bila `|A ∩ token(C)| / |A|` mencapai ambang. Gold ketat selalu ditambahkan. QA dengan kurang dari tiga token jawaban dikeluarkan dari evaluasi multi-gold.

Dari 858 QA, 770 eligible dan 88 dikeluarkan. Hasil sensitivitas adalah:

| Ambang | QA | Total qrels | Rata-rata chunk relevan | Median | Maksimum | QA dengan >1 gold |
|---:|---:|---:|---:|---:|---:|---:|
| 0,5 | 770 | 4.905 | 6,370 | 3 | 83 | 650 |
| 0,6 | 770 | 3.551 | 4,612 | 2 | 53 | 533 |
| 0,7 | 770 | 2.115 | 2,747 | 2 | 25 | 401 |

Kenaikan ambang menurunkan jumlah chunk relevan secara monoton. Label ini merupakan *distant supervision*, bukan pengganti label utama dan tidak divalidasi manual, sesuai batasan penelitian.

## Verifikasi

- Enam unit test untuk normalisasi, determinisme dan disjointness split, token jawaban, rasio overlap, pemaksaan gold ketat, serta pengecualian jawaban pendek lulus.
- Assertion integrasi menghasilkan `STAGE2_ASSERTIONS_OK`.
- Sepuluh split mempunyai nol overlap, cakupan 120 dokumen, dan delapan tipe pada setiap bagian.
- Manifest data identik setelah pipeline dijalankan ulang.

## Reproduksi

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
.\.venv\Scripts\python.exe -m pytest tests\test_data.py tests\test_labels.py -q
```

