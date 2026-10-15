# Audit Tahap 3 — Indeks dan Embedding

Tanggal: 15 Oktober 2026

Status Tahap 3: **SELESAI**

## Komponen sparse

- 13.737 chunk terindeks dalam 120 indeks pasien.
- Koleksi feedback per seed terdiri atas 511–518 QA dan 7.797–8.363 chunk dari tepat 72 dokumen koleksi.
- Semua 10 audit kebocoran berstatus PASS; overlap dokumen dev/test dengan sumber feedback = 0.
- Selisih absolut maksimum terhadap `rank_bm25.BM25Okapi` pada contoh kecil adalah `1,11 × 10^-16`.
- Kebijakan seri: urutan indeks chunk stabil untuk retrieval pasien/eksternal dan acak deterministik per query untuk question similarity; tidak ada peringkat optimistis.

## Benchmark CPU dan keputusan komputasi

| No. | Pendekatan | Throughput | Estimasi konservatif | Hasil |
|---:|---|---:|---:|---|
| 1 | E5-base, CPU, batch 16, sampel 64 | 0,620 item/detik | 28.682 detik (7,97 jam) | Melebihi batas 2 jam |
| 2 | E5-small, CPU, batch 16, sampel 64 | 2,172 item/detik | 8.183 detik (2,27 jam) | Melebihi batas 2 jam |
| 3 | E5-small, CPU, batch 32, sampel 96 | 2,309 item/detik | 7.695 detik (2,14 jam) | Melebihi batas 2 jam |
| 4 | E5-small, deduplikasi, batch 64, sampel 128 | 1,527 item/detik | 10.933 detik (3,04 jam) | Melebihi batas 2 jam |

Karena seluruh estimasi lokal melampaui dua jam, model utama dijalankan pada Kaggle privat.

## Eksekusi Kaggle privat

- Dataset input dan notebook tetap berstatus Private; Internet aktif dan dua GPU Tesla T4 terdeteksi.
- Model `intfloat/multilingual-e5-base` memakai revision `d128750597153bb5987e10b1c3493a34e5a4502a` dengan dimensi 768.
- Encoding 13.708 teks chunk unik, 8 pertanyaan unik, dan 769 jawaban unik selesai dalam 327,239 detik.
- Arsip output 41.582.982 byte diunduh, diekstrak ke `models/embeddings/`, dan sesi GPU dihentikan.
- Dataset Kaggle privat yang memuat teks klinis telah dihapus oleh pengguna setelah hasil lokal dinyatakan lengkap.

## Verifikasi cache lokal

| Berkas | Bentuk | Dtype | Rentang norma L2 | SHA-256 cocok |
|---|---:|---|---:|---|
| `chunks.npy` | 13.737 × 768 | float32 | 0,99999988–1,00000012 | Ya |
| `questions.npy` | 858 × 768 | float32 | 1,00000000–1,00000000 | Ya |
| `answers.npy` | 858 × 768 | float32 | 0,99999994–1,00000012 | Ya |

Seluruh nilai finite; urutan 13.737 `chunk_id` dan 858 `qid` sama persis dengan sumber terproses serta unik. Bukti mesin tersimpan di `results/embeddings/verification.json`.

## Kriteria lolos

- [x] Unit test BM25 dan paritas dengan `rank_bm25` lulus.
- [x] Audit kebocoran sumber feedback lulus untuk seed 1–10.
- [x] Cache E5-base lengkap, ternormalisasi, cocok manifest, dan tersimpan di disk.
