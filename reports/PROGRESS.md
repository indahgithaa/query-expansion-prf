# Progres Penelitian

## Ringkasan pemahaman awal (2026-10-15)
1. Penelitian menguji PRF berbobot sigmoid yang memilih feedback jawaban melalui kemiripan pertanyaan.
2. Retrieval selalu dibatasi pada laporan pasien target; sumber feedback hanya berasal dari bagian koleksi.
3. Delapan tipe pertanyaan dipakai untuk evaluasi dan skenario, tidak sebagai fitur metode.
4. Split berbasis dokumen 60/20/20 diulang untuk seed 1–10 tanpa kebocoran antarbagi.
5. Empat skenario adalah TPL, NEW, PARA, dan PNEW.
6. Pemilihan parameter, ukuran kemiripan, retriever, dan reranker hanya menggunakan dev.
7. Evaluasi utama memakai satu chunk gold ketat; multi-gold hanya metrik pendukung.
8. Baseline, RM3, ablation sumber/bobot, SOFT, komponen semantik, dan oracle harus dipisahkan.
9. Semua klaim angka harus dapat ditelusuri ke artefak privat di `results/`.
10. Data klinis, hasil, model, secret, dan pertukaran Kaggle tidak boleh dilacak Git.
11. Tahap 0–13 dikerjakan berurutan dan setiap tahap harus diverifikasi serta di-commit.
12. Generation bersifat opsional; kondisi login, biaya, atau risiko publikasi data wajib dihentikan.

---

## Checkpoint Tahap 0: Setup (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Menginisialisasi struktur `configs/`, `data/processed/`, `src/`, `scripts/`, `notebooks/`, `results/`, `reports/`, `paper/`, `tests/`, dan `kaggle_io/`.
- Membuat environment `.venv`, memasang seluruh dependensi open-source yang dipin, serta menyiapkan README dan log keputusan.
- Memasang aturan privasi sebelum commit pertama; sumber lokal terdiri atas satu `dataset.json` dan 120 berkas chunk hasil ekstraksi.
- Mengatasi blokir Windows Application Control pada NumPy 2.5.4 dengan build 2.5.3 yang kompatibel.

File yang dibuat:
- `.gitignore`, `requirements.txt`, `README.md`, `reports/PROGRESS.md`, `reports/DECISIONS.md`, dan placeholder struktur direktori.

Bukti verifikasi:
- Python 3.12.10; NumPy 2.5.3; SciPy 1.18.1; pandas 3.0.6; scikit-learn 1.9.1; PyTorch 2.14.1+cpu.
- Uji impor numerik menghasilkan `NUMERIC_IMPORT_OK`; uji impor ML menghasilkan `ML_IMPORT_OK`; CUDA lokal tidak tersedia.
- `pip check`: `No broken requirements found.`
- Seluruh 10 direktori wajib tersedia; `git ls-files data` menghasilkan 0 file; aturan ignore aktif untuk `data/`, `results/`, `models/`, `.env`, cache, `.venv/`, dan `kaggle_io/`.

Kriteria lolos:
- [x] Repository baru dan struktur lengkap.
- [x] Semua versi dependensi langsung dipin dan dapat diimpor.
- [x] Aturan privasi mencegah data serta artefak sensitif masuk Git.

Temuan penting:
- `chunks.zip` tidak tersedia, tetapi 120 file JSON hasil ekstraksinya tersedia dan akan dipakai sebagai sumber ekuivalen.
- Komputasi lokal hanya CPU; estimasi embedding dan reranker akan diukur sebelum memutuskan penggunaan Kaggle.

---

## Checkpoint Tahap 1: Pemahaman Data (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Mengaudit skema, jumlah sumber, distribusi tipe, duplikasi, nama file, join, validitas offset, panjang teks, dan cakupan per dokumen secara read-only.
- Membuat tiga gambar EDA dan menyimpan bukti agregat privat di `results/eda/summary.json`.

File yang dibuat:
- `scripts/run_eda.py`, `reports/01_eda.md`, `reports/01_eda_summary.json`.
- `reports/figures/01_question_type_distribution.png`, `02_answer_length_distribution.png`, dan `03_document_coverage.png`.

Bukti verifikasi:
- 911 QA, 120 dokumen, 13.737 chunk, 8 template; seluruh field wajib lengkap.
- 35 kelompok duplikat dan 53 baris ekstra menghasilkan perkiraan 858 QA setelah deduplikasi.
- Join metadata 911/911; context identik 899/911; hanya 79/911 `answer_start` valid.
- Semua template terpetakan; satu nama mengandung `U+2060`; collision normalisasi 0.
- Assertion otomatis menghasilkan `EDA_ASSERTIONS_OK`; rerun menghasilkan perubahan hash 0 pada lima artefak; ketiga gambar telah diperiksa visual.

Kriteria lolos:
- [x] Statistik dan seluruh masalah data yang diketahui dilaporkan.
- [x] Tiga gambar tersimpan, terbaca, dan dapat direproduksi.

Temuan penting:
- Tipe pedoman hanya tersedia pada 21 dokumen, sehingga pembentukan split harus divalidasi per seed.
- Gold ketat harus memakai kunci metadata dan `answer_start` tidak boleh dipakai.

---

## Checkpoint Tahap 2: Pra-pemrosesan, Split, dan Label (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Menormalisasi nama, menduplikasi ulang secara deterministik, menyatukan QA dengan chunk, memetakan delapan tipe, dan membuat ID stabil.
- Membentuk 10 split per dokumen serta label gold ketat dan multi-gold pada ambang 0,5/0,6/0,7.

File yang dibuat:
- `src/text.py`, `src/data.py`, `src/labels.py`, `scripts/prepare_data.py`, `configs/data.yaml`.
- `tests/test_data.py`, `tests/test_labels.py`, `reports/02_data_preparation.md`.
- Artefak privat di `data/processed/` dan bukti di `results/data_preparation/audit.json`.

Bukti verifikasi:
- Rekonsiliasi lengkap: 911 → 858 setelah deduplikasi → 858 setelah join → 858 final; 0 QA gagal join.
- 120 dokumen dibagi 72/24/24 pada semua seed; overlap 0 dan seluruh 8 tipe hadir pada koleksi/dev/test.
- Gold ketat: 858 qrels; multi-gold eligible: 770 QA, dengan 4.905/3.551/2.115 qrels pada ambang 0,5/0,6/0,7.
- `6 passed`, `STAGE2_ASSERTIONS_OK`, dan manifest deterministik pada rerun.

Kriteria lolos:
- [x] Rekonsiliasi jumlah lengkap dan seluruh QA final berhasil join.
- [x] Sepuluh split bebas tumpang tindih dan lengkap tipe.
- [x] Label ketat dan tiga label multi-gold tersimpan serta sensitivitas dilaporkan.

Temuan penting:
- 28 dari 35 kelompok duplikat menunjuk lokasi gold berbeda; kebijakan ID terkecil dibekukan sebelum retrieval.
- Label multi-gold sensitif terhadap ambang, sehingga ketiga ambang wajib dilaporkan terpisah.

---

## Checkpoint Tahap 3: Indeks dan Sumber Feedback (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Membuat 120 indeks BM25 pasien dan 10 pasang indeks pertanyaan/chunk feedback, dengan pemecahan seri stabil serta audit kebocoran.
- Menjalankan E5-base di notebook Kaggle privat setelah benchmark CPU melampaui dua jam.
- Mengunduh, mengekstrak, dan memverifikasi cache embedding; menghentikan sesi GPU dan menghapus dataset privat setelah verifikasi.
- Menyelaraskan notebook lokal dengan tata letak input Kaggle yang sebenarnya dan menambah verifier cache yang dapat dijalankan ulang.

File yang dibuat/diubah:
- `src/bm25.py`, `src/sources.py`, `src/dense.py`, `scripts/build_indices.py`, `scripts/build_embeddings.py`, `scripts/verify_embeddings.py`.
- `scripts/prepare_kaggle_embeddings.py`, `notebooks/run_embeddings.ipynb`, pengujian Tahap 3, dan `reports/03_embedding_attempts.md`.
- Artefak privat `models/indices/`, `models/embeddings/`, `results/indices/`, dan `results/embeddings/`.

Bukti verifikasi:
- 13.737 chunk dalam 120 indeks pasien; sumber feedback seed 1–10 hanya berisi 72 dokumen koleksi dan overlap dev/test = 0.
- Selisih maksimum implementasi BM25 terhadap `rank_bm25` adalah `1,11 × 10^-16`.
- E5-base revision `d128750597153bb5987e10b1c3493a34e5a4502a`; waktu encoding GPU 327,239 detik.
- Bentuk embedding: chunk `(13737, 768)`, pertanyaan `(858, 768)`, jawaban `(858, 768)`; dtype float32, seluruh nilai finite dan norma L2 ≈ 1.
- Semua hash SHA-256 cocok manifest dan urutan ID cocok sumber; seluruh test lokal lulus.

Kriteria lolos:
- [x] Unit test BM25 lulus dan paritas pustaka terverifikasi.
- [x] Uji kebocoran lulus untuk seluruh pengulangan.
- [x] Cache embedding tersimpan dan terverifikasi.

Temuan penting:
- Komputasi GPU memang diperlukan: estimasi CPU E5-base 7,97 jam, sedangkan Kaggle T4 menyelesaikan encoding dalam sekitar 5,45 menit.
- Kaggle mengekstrak arsip dataset secara otomatis; loader notebook kini mencari tepat satu `chunks.jsonl` dan `qa.jsonl` secara rekursif.

---

## Checkpoint Tahap 4: Set Parafrasa dan Skenario (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Menulis enam parafrasa dev dan enam parafrasa test untuk masing-masing dari delapan tipe pertanyaan.
- Membuat assignment parafrasa deterministik per `(seed, qid)` dan manifest TPL/NEW/PARA/PNEW untuk 10 pengulangan.
- Mengimplementasikan leave-one-type-out untuk koleksi QA pada NEW/PNEW; korpus eksternal tidak berubah.

File yang dibuat:
- `src/scenarios.py`, `scripts/prepare_paraphrases.py`, `scripts/embed_paraphrases.py`, `configs/scenarios.yaml`, dan `tests/test_scenarios.py`.
- `reports/04_paraphrases.md`; artefak privat `data/processed/paraphrases.json`, `data/processed/scenarios/`, dan `results/scenarios/audit.json`.

Bukti verifikasi:
- 96 parafrasa: 48 dev dan 48 test; tiap tipe tepat 6+6.
- Duplikasi internal 0, overlap dev–test 0, identik dengan template 0, dan seluruh pemeriksaan penanda maksud lulus.
- 6.830 pemeriksaan NEW/PNEW lintas 10 seed lulus tanpa satu pun QA bertipe target di koleksi pertanyaan.
- 17 unit test lulus.

Kriteria lolos:
- [x] Seluruh 96 parafrasa tersimpan dan ditampilkan dalam laporan.
- [x] Pemeriksaan otomatis identitas, overlap, dan maksud lulus.
- [x] Skenario NEW/PNEW bebas QA dengan tipe sama.

---

## Checkpoint Tahap 5: Baseline, PRF Klasik, dan Metode Inti BM25 (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Mengimplementasikan min-max, sigmoid stabil, RRF, query feedback berbobot, softmax, dan RM3.
- Menjalankan B0, B1, RM3, P-EXT, P-QA, P-FUS, ZONE3, SOFT, SOFT-SM, dan ORACLE-W pada dev seed 1–3.
- Menyetel grid `(m,r)`, lalu `k,tau`, dengan kendala tidak ada skenario turun lebih dari 0,01 terhadap konfigurasi awal.

File yang dibuat:
- `src/methods.py`, `scripts/run_stage5.py`, `tests/test_methods.py`, `configs/stage5.yaml`, dan `reports/05_bm25_methods.md`.
- `reports/figures/05_soft_heatmap.png` serta artefak privat di `results/stage5/`.

Bukti verifikasi:
- 2.052 kombinasi query–skenario dev dievaluasi; parameter dasar terpilih `m=5, r=3`.
- SOFT dev: TPL 0,6133; NEW 0,3587; PARA 0,3577; PNEW 0,2826; rata-rata 0,4031.
- B1 rata-rata 0,2442; RM3 terbaik (`5,30,0,5`) 0,2560; ORACLE-W 0,4800.
- Selisih terhadap sanity check hanya 0,0208 untuk B1 dan 0,0121 untuk SOFT.
- Heatmap diperiksa visual dan 21 unit test lulus sebelum run.

Kriteria lolos:
- [x] Seluruh konfigurasi berbasis BM25 berjalan di dev.
- [x] Parameter dasar dibekukan hanya dari dev.
- [x] Heatmap tersedia.
- [x] Unit test rumus dan tie-breaking lulus.

---

## Checkpoint Tahap 6: Pemilihan Komponen Semantik di Dev (2026-10-15)

Status: **SELESAI**

Yang dikerjakan:
- Menilai QS-BM25, QS-EMB, dua varian QS-HYB, retriever BM25, dense teks penuh, dense Rocchio, dan seluruh pasangan hybrid yang ditentukan protokol pada dev seed 1–3.
- Membuat cache embedding teks penuh E5-base dan cache reranker cross-encoder di Kaggle privat, lalu memverifikasi revision, jumlah baris, nilai finite, dan hash hasil secara lokal.
- Menyetel lambda reranker untuk B1+RR dan kandidat FINAL dengan kendala penurunan maksimum 0,01 per skenario.
- Mengunci konfigurasi terpilih di `configs/final.yaml` sebelum Tahap 7.

File yang dibuat/diubah:
- `src/dense.py`, `src/semantic.py`, `src/reranker.py`, dan `src/methods.py`.
- `scripts/run_stage6_selection.py`, skrip persiapan/impor cache Kaggle, `scripts/select_stage6_reranker.py`, dan dua notebook Kaggle Tahap 6.
- `configs/stage6_pre_reranker.yaml`, `configs/final.yaml`, `tests/test_semantic.py`, dan `reports/06_selection.md`.
- Artefak privat di `models/embeddings/`, `models/reranker/`, `results/stage6/`, dan `kaggle_io/`.

Bukti verifikasi:
- Seleksi hanya memakai 2.052 observasi query–skenario dev seed 1–3; audit eksplisit mencatat `test_used_for_selection=false`.
- QS-BM25 dan R-BM25 adalah satu-satunya pilihan yang memenuhi kendala per-skenario; konfigurasi sebelum reranker memiliki MRR rata-rata 0,4031.
- Cache reranker berisi 61.011 pasangan unik, memakai `BAAI/bge-reranker-v2-m3` tanpa fallback, dan lolos verifikasi hash.
- Sesi GPU dihentikan dan kedua dataset privat sementara Kaggle dihapus setelah hasil lokal terverifikasi.
- FINAL memakai top-20 dan lambda 0,25: TPL 0,6155; NEW 0,3698; PARA 0,3792; PNEW 0,3037; rata-rata 0,4171.
- B1+RR memakai lambda 0,75 dengan rata-rata 0,3183.
- Seluruh unit test lokal lulus setelah implementasi.

Kriteria lolos:
- [x] Cache skor reranker lengkap dan model utama berhasil digunakan.
- [x] Langkah pemilihan 2–4 dijalankan hanya pada dev seed 1–3.
- [x] Alasan penerimaan dan penolakan tiap komponen terdokumentasi.
- [x] `reports/06_selection.md` dan `configs/final.yaml` tersedia sebelum Tahap 7.

---

## Checkpoint Tahap 7: Run Final 10 Pengulangan (2026-10-16)

Status: **SELESAI**

Yang dikerjakan:
- Menyetel `k`, `tau`, lambda B1+RR, dan lambda FINAL secara terpisah pada dev setiap seed 1–10.
- Mengevaluasi 16 konfigurasi wajib pada test di skenario TPL, NEW, PARA, dan PNEW setelah parameter seed terkunci.
- Menyimpan ranking maksimal top-50, rank per query, grid parameter, serta hash setiap artefak.
- Menambahkan verifier independen untuk cakupan, integritas referensial, posisi gold, dan pemisahan dev/test.

File yang dibuat:
- `scripts/run_stage7.py`, `scripts/verify_stage7.py`, `tests/test_stage7.py`, dan `reports/07_final_runs.md`.
- Artefak privat di `results/runs/`, `results/params/`, dan `results/stage7/`.

Bukti verifikasi:
- 10 seed, 6.888 observasi query–skenario test, 16 metode, dan 110.208 baris ranking selesai.
- Dua puluh hash output cocok; setiap query memiliki seluruh metode tanpa ranking duplikat atau chunk dari pasien lain.
- Posisi gold pada ranking cocok dengan seluruh kolom rank; empat dokumen dengan kurang dari 50 chunk ditangani dengan menyimpan seluruh chunk.
- Semua file parameter mencatat `selection_partition=dev`, `evaluation_partition=test`, dan `test_used_for_tuning=false`.
- 29 unit test lulus sebelum verifikasi run lengkap.

Kriteria lolos:
- [x] Parameter adaptif dipilih ulang hanya dari dev di setiap pengulangan.
- [x] Seluruh konfigurasi wajib dievaluasi pada empat skenario test.
- [x] Ranking top-50 dan parameter tersedia untuk seluruh seed.
- [x] Audit lengkap lulus tanpa pelanggaran kebocoran test.





