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



