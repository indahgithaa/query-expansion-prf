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

## Status Tahap 3: TERHENTI (2026-10-15)

Tahap belum memenuhi kriteria karena cache embedding belum tersedia. Komponen sparse selesai dan terverifikasi: 120 indeks pasien, 10 pasang indeks feedback, audit kebocoran PASS untuk seed 1–10, perbedaan BM25 terhadap `rank_bm25` sebesar `1,11 × 10^-16`, dan 11 test lulus.

Embedding E5-base CPU diperkirakan 7,97 jam. Tiga konfigurasi model cadangan E5-small tetap melampaui batas dua jam (2,27; 2,14; dan 3,04 jam). Kaggle tidak dapat dibuka karena computer-use mengembalikan inventaris aplikasi/browser kosong; Chrome, peluncuran Chrome, dan IAB semuanya tidak tersedia. Detail lengkap ada di `reports/03_embedding_attempts.md`.

Sesuai kondisi berhenti untuk tahap wajib setelah tiga upaya perbaikan, pekerjaan dihentikan sampai browser yang dapat dikendalikan tersedia dan sesi Kaggle pengguna sudah login.

### Pembaruan persiapan Kaggle

Tanpa melangkahi kriteria Tahap 3, paket input privat dan notebook eksekusi telah disiapkan. `embedding_inputs.zip` berukuran 3.415.577 byte dan memuat tepat data olahan yang dibutuhkan. `notebooks/run_embeddings.ipynb` memiliki tujuh sel dengan guard GPU/internet, E5-base, validasi embedding, hash, serta arsip output. Pemeriksaan menghasilkan `13 passed` dan `KAGGLE_PACKAGE_OK`. Tahap tetap belum lolos sampai hasil benar-benar dijalankan, diunduh, dan diverifikasi.





