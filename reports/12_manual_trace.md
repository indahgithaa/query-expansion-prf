# Tahap 12 — Jejak Perhitungan Manual

## Pemilihan contoh

Contoh dipilih secara deterministik sebagai kenaikan B1→SOFT terbesar dari 20 observasi test yang dokumennya memiliki kurang dari 40 chunk. Hasilnya adalah seed **7**, `Q0100` skenario **TPL** pada `D0022` (32 chunk). Chunk gold `D0022:P0010:C001` naik dari rank 14 menjadi 1; FINAL menempatkannya pada rank 1.

## 1. Tokenisasi pertanyaan

> Bagaimana menyusun diagnosis gizi dengan format PES pada pasien ini?

- Sebelum penghapusan stopword: `['bagaimana', 'menyusun', 'diagnosis', 'gizi', 'dengan', 'format', 'pes', 'pada', 'pasien', 'ini']`
- Setelah penghapusan stopword (`q'`): `['menyusun', 'diagnosis', 'gizi', 'format', 'pes']`

Tokenisasi memakai lowercase dan regex `\w+`. Daftar Sastrawi ditambah `apa`, `bagaimana`, `yang`, `ini`, dan `pasien`.

## 2. Question similarity QS-BM25

Indeks pertanyaan koleksi berisi N=515 pertanyaan dengan `avgdl=8,460194`. QS memakai token sebelum stopword dihapus. IDF negatif mengikuti implementasi `rank_bm25`: nilainya diganti dengan `epsilon × average_IDF`, sehingga beberapa term umum memiliki IDF positif yang sama.

| Term | df | IDF yang dipakai |
|---|---:|---:|
| `bagaimana` | 359 | 0,363448 |
| `menyusun` | 72 | 1,811111 |
| `diagnosis` | 72 | 1,811111 |
| `gizi` | 216 | 0,324524 |
| `dengan` | 144 | 0,944269 |
| `format` | 72 | 1,811111 |
| `pes` | 72 | 1,811111 |
| `pada` | 288 | 0,363448 |
| `pasien` | 443 | 0,363448 |
| `ini` | 371 | 0,363448 |

Tiga pertanyaan feedback teratas:

| Rank | QID | Panjang | Skor | Kontribusi term terbesar |
|---:|---|---:|---:|---|
| 1 | `Q0651` | 10 | 9,212502 | diagnosis=1,674006; format=1,674006; menyusun=1,674006; pes=1,674006 |
| 2 | `Q0376` | 10 | 9,212502 | diagnosis=1,674006; format=1,674006; menyusun=1,674006; pes=1,674006 |
| 3 | `Q0209` | 10 | 9,212502 | diagnosis=1,674006; format=1,674006; menyusun=1,674006; pes=1,674006 |

Contoh lengkap term `menyusun` pada pertanyaan rank 1:

- IDF mentah = `ln((515 - 72 + 0,5) / (72 + 0,5)) = 1,811111`; IDF yang dipakai = `1,811111`.
- tf=1, dl=10, avgdl=8,460194, k1=1,5, b=0,75.
- Penyebut = `tf + k1 × (1-b + b × dl/avgdl) = 2,704757`.
- Kontribusi = `IDF × tf × (k1+1) / penyebut = 1,674006`.
- Skor similarity maksimum `s = 9,212502` (jumlah kontribusi seluruh term query pada pertanyaan rank 1).

## 3. Feedback dan susunan query ekspansi

Parameter seed 7 adalah `m=5` dan `r=3`. Lima jawaban feedback QA:

| Rank | QID | Skor QS | Token | Kutipan dipersingkat |
|---:|---|---:|---:|---|
| 1 | `Q0651` | 9,212502 | 106 | - NI-2.11. Keterbatasan penerimaan makan berkaitan dengan reflek orofaring belum kuat ditandai dengan pasien terpasang OGT - NI-1.1 Peningkatan Kebut… |
| 2 | `Q0376` | 9,212502 | 116 | NI-2.11 Keterbatasan penerimaan makanan berkaitan dengan adanya gangguan pada saluran pernapasan ditandai dengan hasil recall energi (53,49%), protei… |
| 3 | `Q0209` | 9,212502 | 95 | NI – 2.9 Keterbatasan penerimaan makanan berkaitan dengan kondisi pasien sesak, terdapat pendarahan pada gastrointestinal dan adanya ganggan tumbuh k… |
| 4 | `Q0296` | 9,212502 | 77 | NI-2.9 Keterbatasan Penerimaan Makanan berkaitan dengan memasuki fase ebb pasca operasi ditandai dengan diagnose medis post explorasi laparotomy hari… |
| 5 | `Q0024` | 9,212502 | 113 | ▪ NI-2.1 Asupan makan dan minum peroral tidak adekuat berkaitan dengan keadaan umum lemah dan penurunan nafsu makan serta status gizi kurang ditandai… |

Lima chunk feedback eksternal:

| Rank | Chunk | Token | Kutipan dipersingkat |
|---:|---|---:|---|
| 1 | `D0041:P0019:C001` | 91 | 021). 2.2 Diagnosis Gizi Pasien didiagnosis menderita penyakit Spina bifida, Microcephali + Colpocephaly, Ruptured Meningomyelocele Lumbosacral, Arno… |
| 2 | `D0117:P0021:C000` | 93 | BAB 2 RENCANA INTERVENSI DAN MONEV 2.1 Diagnosis Gizi Penegakan diagnosis gizi dilakukan untuk mengidentifikasi masalah gizi, faktor penyebab yang me… |
| 3 | `D0041:P0019:C002` | 98 | (3) mengidentifikasi etiologi (penyebab masalah), dan (4) menyusun pernyataan diagnosis gizi menggunakan format PES (Problem-Etiology- Signs and Symp… |
| 4 | `D0053:P0062:C000` | 91 | LEMBAR KONSULTASI MAHASISWA Sub-rotasi : IKA- Gizi Klinik Lokasi : RSUD Sidoarjo Nama : [disamarkan] [TABLE 0] No \| Tanggal \| Materi Konsultasi \|… |
| 5 | `D0052:P0063:C001` | 58 | metri pada pasien anak dengan kegawatan What practices I want to continue? What areas I wish to improve in? • Implementasi manajemen bangsal yang • K… |

Susunan tepat secara konseptual: `q_QA = [menyusun diagnosis gizi format pes] × 3 + jawaban Q0651 + Q0376 + Q0209 + Q0296 + Q0024`; `q_EXT = [menyusun diagnosis gizi format pes] × 3 + chunk D0041:P0019:C001 + D0117:P0021:C000 + D0041:P0019:C002 + D0053:P0062:C000 + D0052:P0063:C001`.

Implementasi BM25 memakai bag-of-words berbobot: setiap kemunculan term query dasar ditambah bobot 3, lalu seluruh kemunculan token feedback ditambahkan dengan bobot 1. Representasi lengkap tersimpan di artefak privat `results/stage12/manual_trace.json`.

## 4. BM25 baseline B1

Indeks pasien mempunyai N=32 chunk dan `avgdl=113,656250`. Tiga chunk teratas:

| Rank | Chunk | Panjang | Skor B1 | Kontribusi term |
|---:|---|---:|---:|---|
| 1 | `D0022:P0020:C000` | 94 | 4,222960 | diagnosis=3,301459; gizi=0,921501 |
| 2 | `D0022:P0005:C000` | 138 | 1,180460 | gizi=1,180460 |
| 3 | `D0022:P0005:C001` | 76 | 1,107015 | gizi=1,107015 |

Contoh lengkap term `diagnosis` pada chunk rank 1:

- IDF mentah = `ln((32 - 1 + 0,5) / (1 + 0,5)) = 3,044522`; IDF dipakai `3,044522`.
- tf=1, dl=94, avgdl=113,656250, penyebut=2,305437.
- Kontribusi = `3,044522 × 1 × (1,5+1) / 2,305437 = 3,301459`.
Chunk gold memiliki skor B1 `0,644014` dan rank **14**.

## 5. Normalisasi min–max

Normalisasi dilakukan atas seluruh 32 chunk: QA min/max = `19,060102`/`412,671747`, EXT min/max = `11,359375`/`113,811298`. Rumusnya `x̂=(x-min)/(max-min)`.

| Chunk (top-3 SOFT) | QA raw | QA norm | EXT raw | EXT norm |
|---|---:|---:|---:|---:|
| `D0022:P0010:C001` | 412,671747 | 1,000000 | 106,304771 | 0,926731 |
| `D0022:P0006:C000` | 164,183667 | 0,368697 | 81,088034 | 0,680599 |
| `D0022:P0019:C000` | 151,964378 | 0,337653 | 100,723011 | 0,872249 |

## 6. Bobot sigmoid dan skor SOFT

Dengan `k=1,0`, `τ=3,577024`, dan `s=9,212502`:

`w(s) = 1 / (1 + exp(-1,0 × (9,212502 - 3,577024))) = 0,996444`.

| Chunk | Perhitungan SOFT | Skor SOFT |
|---|---|---:|
| `D0022:P0010:C001` | `0,996444×1,000000 + 0,003556×0,926731` | 0,999739 |
| `D0022:P0006:C000` | `0,996444×0,368697 + 0,003556×0,680599` | 0,369807 |
| `D0022:P0019:C000` | `0,996444×0,337653 + 0,003556×0,872249` | 0,339554 |

Chunk gold menjadi rank **1** pada SOFT.

## 7. Reranker FINAL

FINAL mererank top-20 kandidat SOFT dengan `BAAI/bge-reranker-v2-m3` dan λ=0,25. Skor pipeline dan reranker dinormalisasi min–max lagi di dalam 20 kandidat, lalu digabungkan sebagai `FINAL = λ × reranker_norm + (1-λ) × pipeline_norm`.

Rentang kandidat: pipeline `0,125789`–`0,999739`; reranker raw `-6,851562`–`-2,423828`.

| Rank FINAL | Chunk | Pipeline norm | Reranker raw | Reranker norm | Skor gabungan |
|---:|---|---:|---:|---:|---:|
| 1 | `D0022:P0010:C001` | 1,000000 | -3,230469 | 0,817821 | 0,954455 |
| 2 | `D0022:P0006:C000` | 0,279212 | -3,337891 | 0,793560 | 0,407799 |
| 3 | `D0022:P0019:C000` | 0,244596 | -3,302734 | 0,801500 | 0,383822 |

Chunk gold tetap pada rank **1** setelah reranking.

## 8. Dampak pada metrik

Dengan satu gold ketat, `RR=1/rank`. Jika rank≤10, `nDCG@10=1/log2(rank+1)`; jika rank>10 nilainya 0.

| Metode | Rank gold | Reciprocal rank | nDCG@10 |
|---|---:|---:|---:|
| B1 | 14 | 0,071429 | 0,000000 |
| SOFT | 1 | 1,000000 | 1,000000 |
| FINAL | 1 | 1,000000 | 1,000000 |

Jadi, untuk contoh ini SOFT menaikkan RR dari 0,071429 menjadi 1,000000 dan nDCG@10 dari 0 menjadi 1. Reranker tidak mengubah rank gold lebih lanjut.

## 9. Contoh Robustness Index

Dari tabel pooled FINAL vs B1 Tahap 8: win=4268, tie=963, loss=1657, N=6888. Tie tetap masuk penyebut tetapi tidak masuk pembilang:

`RI = (N⁺ - N⁻) / N = (4268 - 1657) / 6888 = 0,379065`.

Nilai positif berarti FINAL lebih sering memperbaiki daripada memperburuk rank gold relatif terhadap B1.

## Verifikasi

Semua ranking B1, SOFT, dan FINAL direkonstruksi dari fitur serta cache skor, kemudian dicocokkan persis dengan ranking beku Tahap 7.
SHA-256 artefak trace privat: `b3d2cd99b4928287348a7843b9521c5bcf845678a3ea64e169e39adbb1d19778`.
