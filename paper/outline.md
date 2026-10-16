# Outline Paper Jurnal

## Judul sementara

**Adaptive Pseudo-Relevance Feedback with Question Similarity for Clinical Nutrition Report Retrieval**

## Abstrak terstruktur

- **Latar belakang:** pertanyaan asuhan gizi pendek dan generik menyulitkan pencarian bagian laporan yang tepat.
- **Tujuan:** membandingkan PRF klasik dengan PRF dua sumber yang bobotnya ditentukan question similarity.
- **Metode:** 858 QA, 120 laporan, sepuluh split dokumen, empat skenario, evaluasi strict dan multi-gold.
- **Hasil utama:** FINAL MRR 0,3961 vs B1 0,2478 dan RM3 0,2551; delta FINAL–B1 0,1483, CI [0,1405; 0,1559]. Sumber: `results/stage8/strict_summary.csv` dan `statistics_combined.csv`.
- **Kesimpulan:** feedback QA membantu bila pertanyaan serupa tersedia, sedangkan pembobotan adaptif membatasi kerugian pada topik baru.

## 1. Pendahuluan

1. Masalah retrieval pada laporan panjang dan query generik.
2. Kelemahan PRF lokal ketika initial retrieval lemah.
3. Hipotesis: pertanyaan serupa menyediakan feedback yang lebih terarah; similarity dapat menjadi sinyal kepercayaan.
4. Kontribusi:
   - formulasi dua sumber feedback dengan sigmoid;
   - evaluasi empat kondisi availability/paraphrase;
   - pemilihan komponen dengan kendala ketahanan;
   - analisis statistik, komponen, dan failure modes.

Rujukan konseptual harus diverifikasi dari sumber asli sebelum submission.

## 2. Pekerjaan terkait

- BM25 dan pseudo-relevance feedback.
- Collection enrichment dan selective query expansion.
- Question retrieval/question similarity.
- Dense dan hybrid retrieval, RRF, cross-encoder reranking.
- Retrieval-augmented generation untuk domain klinis.

## 3. Metode

### 3.1 Data dan privasi

- 911 QA mentah → 858 setelah deduplikasi/join; 13.737 chunk dan 120 dokumen.
- Sumber: `results/data_preparation/audit.json`.
- Satu gold utama; multi-gold distant supervision sebagai sensitivitas.

### 3.2 Desain evaluasi

- Split dokumen 60/20/20, seed 1–10.
- TPL, NEW, PARA, PNEW.
- Tipe pertanyaan hanya dipakai untuk skenario/evaluasi, bukan fitur metode.
- Sumber: `data/processed/splits/`, `results/stage7/audit.json`.

### 3.3 Baseline dan PRF klasik

- B1, RM3, dense/hybrid, dan reranker baseline.
- RM3: 5 feedback docs, 30 terms, interpolasi 0,5.
- Sumber: `configs/stage5.yaml`.

### 3.4 Metode adaptif

- QS-BM25; feedback QA dan eksternal; $m=5$, $r=3$.
- Min–max, sigmoid, dan kombinasi SOFT.
- Cross-encoder opsional pada top-20.
- Sumber: `configs/final.yaml`.

### 3.5 Pemilihan dev

- Pemilihan berurutan dengan batas penurunan per skenario 0,01.
- Test tidak dipakai untuk tuning.
- Sumber: `results/stage6/selection_reranker.json` dan `results/params/`.

### 3.6 Metrik dan statistik

- MRR, Recall@k, nDCG@10, MAP@50 multi-gold.
- RI, Wilcoxon, Holm, bootstrap 10.000, rank-biserial.

## 4. Hasil

### 4.1 Hasil utama

Masukkan `paper/tables/main_results.tex` dan `paper/figures/main_mrr_comparison.pdf`.

Klaim:

- FINAL terbaik di antara metode non-oracle: MRR 0,3961, R@10 0,7206, nDCG@10 0,4650. Sumber: `results/stage8/strict_summary.csv`.
- RM3 hanya 0,2551. Sumber sama.
- ORACLE-W 0,4659 menunjukkan ruang 0,0699 MRR. Sumber sama.

### 4.2 Signifikansi dan robustness

Masukkan `paper/tables/statistical_comparisons.tex`.

- FINAL–B1 +0,1483 [0,1405; 0,1559], signifikan 10/10 seed. Sumber: `results/stage8/statistics_combined.csv`.
- RI FINAL 0,3791; 4.268/963/1.657 W/T/L. Sumber: `results/stage8/robustness_summary.csv`.

### 4.3 Komponen

Masukkan `paper/tables/component_decomposition.tex` dan gambar similarity.

- Ekspansi/pembobotan 93,97%; reranker 6,03%. Sumber: `results/stage9/decomposition.csv`.
- QS/dense semantik menaikkan beberapa skenario tetapi gagal kendala ketahanan. Sumber: `results/stage9/question_similarity.csv` dan `retriever_analysis.csv`.

### 4.4 Analisis tipe dan kegagalan

- Tujuan diet tertinggi; biokimia terendah; PES memperoleh delta terbesar.
- Gap oracle terbesar di PARA.
- 1.926/6.888 observasi FINAL gagal top-10.
- Sumber: `results/stage10/`.

## 5. Pembahasan

1. Mengapa RM3 lokal terbatas.
2. Mengapa QA feedback kuat pada TPL/PARA tetapi berisiko pada NEW/PNEW.
3. Peran sigmoid sebagai pengendali risiko.
4. Mengapa reranker membantu baseline lebih besar daripada SOFT.
5. Implikasi gap oracle dan failure analysis.
6. Jangan mengklaim dampak generation karena Tahap 11 SKIPPED.

## 6. Keterbatasan dan etika

Ringkas `paper/limitations_and_ethics.md`. Tekankan data sensitif, gold tunggal, multi-gold otomatis, parafrasa sintetis/terkontrol, dan belum adanya evaluasi generatif/manusia.

## 7. Kesimpulan

PRF dua sumber dengan pembobotan question similarity memberikan kenaikan besar dan konsisten atas BM25 dan RM3. Komponen leksikal yang stabil dipilih daripada alternatif semantik yang tidak memenuhi kendala skenario. Evaluasi eksternal dan generation tetap diperlukan.

## Checklist sebelum submission

- Verifikasi seluruh detail bibliografis dari sumber asli.
- Sesuaikan format tabel/gambar dengan template jurnal.
- Pastikan tidak ada data privat di arsip submission.
- Tambahkan pernyataan etik/institutional approval sesuai kebijakan institusi.
- Jalankan verifier dan catat commit/release hash.

