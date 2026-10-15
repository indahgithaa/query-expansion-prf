# Audit Percobaan Tahap 3 — Embedding

Tanggal: 15 Oktober 2026

Status Tahap 3: **TERHENTI — MEMERLUKAN AKSES BROWSER KAGGLE**

Komponen sparse telah selesai: 120 indeks BM25 per laporan, 10 indeks pertanyaan, 10 indeks chunk eksternal, pembandingan `rank_bm25`, dan audit kebocoran. Cache embedding wajib belum selesai.

## Bukti komponen sparse

- 13.737 chunk terindeks dalam 120 indeks pasien.
- Koleksi feedback per seed terdiri atas 511–518 QA dan 7.797–8.363 chunk dari tepat 72 dokumen koleksi.
- Semua 10 audit kebocoran berstatus PASS; overlap dokumen dev/test dengan sumber feedback = 0.
- Selisih absolut maksimum terhadap `rank_bm25.BM25Okapi` pada contoh kecil adalah `1,11 × 10^-16`.
- Kebijakan seri: urutan indeks chunk stabil untuk retrieval pasien/eksternal dan acak deterministik per query untuk question similarity; tidak ada peringkat optimistis.
- Sebelas unit test keseluruhan lulus.

## Percobaan embedding

| No. | Pendekatan | Throughput | Estimasi konservatif | Hasil |
|---:|---|---:|---:|---|
| 1 | E5-base, CPU, batch 16, sampel 64 | 0,620 item/detik | 28.682 detik (7,97 jam) | Gagal batas 2 jam |
| 2 | E5-small cadangan, CPU, batch 16, sampel 64 | 2,172 item/detik | 8.183 detik (2,27 jam; estimator awal) | Gagal batas 2 jam |
| 3 | E5-small, CPU, batch 32, sampel 96 | 2,309 item/detik | 7.695 detik (2,14 jam; estimator awal) | Gagal batas 2 jam |
| 4 | E5-small, deduplikasi teks, batch 64, sampel 128 | 1,527 item/detik | 10.933 detik (3,04 jam) | Gagal batas 2 jam |

Estimator pada percobaan 2–3 masih menghitung QA berulang. Kode kemudian diperbaiki agar hanya menghitung 8 pertanyaan unik dan 769 jawaban unik (total 14.514 teks bersama chunk). Percobaan 4 memakai estimator yang sudah benar, tetapi batch besar menurunkan throughput sehingga tetap gagal.

## Percobaan Kaggle melalui computer-use

Computer-use diinisialisasi sesuai skill. Inventaris mengembalikan `apps: []` dan `browsers: []`. Tiga jalur dicoba:

1. Membuka tab melalui browser `chrome` → `Browser is not available: chrome`.
2. Meluncurkan Chrome → API peluncuran tidak tersedia pada runtime tersebut.
3. Membuka browser bawaan `iab` → `Browser is not available: iab`.

Tidak ada halaman Kaggle yang berhasil dibuka, sehingga tidak ada login, 2FA, captcha, unggahan data, atau perubahan eksternal yang terjadi.

## Kebutuhan dari pengguna

Buka/aktifkan browser yang dapat dikendalikan oleh Codex (idealnya Chrome), pastikan sesi Kaggle sudah login, lalu minta penelitian dilanjutkan. Setelah browser terdeteksi, cache embedding akan dihitung pada notebook/dataset Kaggle privat dan hasilnya diunduh serta diverifikasi. Jika Kaggle meminta login, 2FA, verifikasi telepon, atau captcha, pengguna perlu mengambil alih sesuai protokol.

## Persiapan Kaggle yang telah selesai

Walaupun browser belum tersedia, seluruh artefak lokal telah disiapkan agar eksekusi UI tidak membutuhkan perubahan metodologis:

- `kaggle_io/embeddings/embedding_inputs.zip` berukuran 3.415.577 byte, SHA-256 `82a5317411e997e976923e3421140b9612fd6857e39b54f55237d5858769cfb7`.
- Arsip berisi tepat `chunks.jsonl` (13.737 chunk) dan `qa.jsonl` (858 QA); arsip dan manifest tetap diabaikan Git.
- `notebooks/run_embeddings.ipynb` memiliki tujuh sel: pemeriksaan Python/GPU/internet, instalasi dependensi pin, ekstraksi input, E5-base dengan revision tercatat, encoding GPU, validasi shape/finite/norma L2, SHA-256 output, dan pembuatan `e5_base_embeddings.zip`.
- Dataset dan notebook diwajibkan berstatus Private; dataset harus dihapus setelah hasil terunduh dan diverifikasi.
- Tiga belas test lokal lulus dan pemeriksaan integrasi menghasilkan `KAGGLE_PACKAGE_OK`.

