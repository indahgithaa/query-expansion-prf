# Query Expansion PRF dengan Question Similarity

Implementasi penelitian **Penerapan Query Expansion Berbasis Pseudo-Relevance Feedback dengan Question Similarity pada Retrieval-Augmented Generation Asuhan Gizi Klinis**.

Pipeline mengevaluasi BM25, RM3, feedback pertanyaan serupa, feedback korpus eksternal, pembobotan sigmoid, dense/hybrid retrieval, dan cross-encoder reranking pada empat skenario. FINAL mencapai MRR 0,3961 dibanding B1 0,2478 pada 10 split dokumen. Ringkasan lengkap tersedia di [reports/FINAL_REPORT.md](reports/FINAL_REPORT.md).

## Privasi

Data laporan tetap sensitif walaupun sudah disamarkan. Jangan mengunggahnya ke repository, issue, atau dataset publik. `.gitignore` mengecualikan `data/`, `results/`, `models/`, `kaggle_io/`, kredensial, dan cache. Jika memakai Kaggle, dataset wajib **Private** dan harus dihapus setelah hasil lokal terverifikasi.

## Struktur

- `src/`: BM25, metode PRF, semantic retrieval, reranker, label, dan metrik.
- `scripts/`: persiapan, eksperimen, evaluasi, verifikasi, serta runner.
- `configs/`: parameter terpilih; `final.yaml` dikunci sebelum test.
- `notebooks/`: notebook GPU Kaggle.
- `reports/`: laporan setiap tahap, draf Bab III–IV, dan laporan akhir.
- `paper/`: outline, tabel LaTeX/CSV, gambar PNG/PDF, etika, dan reproduksibilitas.
- `tests/`: unit test dan integration-style checks.

Folder privat yang harus tersedia secara lokal adalah `data/`, `results/`, `models/`, dan `kaggle_io/`.

## Instalasi

Lingkungan referensi lokal menggunakan Windows 11 dan Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

NumPy dipin ke 2.5.3 karena build 2.5.4 diblokir Windows Application Control pada mesin referensi.

## Data input

Tempatkan input pada:

```text
data/dataset/dataset.json
data/chunks/*.chunks.json
```

Pipeline mengharapkan sekitar 911 QA mentah, 120 berkas laporan, dan 13.737 chunk. Jangan menjalankan ulang menggunakan input berbeda tanpa meninjau rekonsiliasi.

## Menjalankan pipeline

Runner dibagi menjadi empat fase agar langkah GPU privat tidak dijalankan tanpa sengaja.

```powershell
.\scripts\run_all.ps1 -Phase prepare
.\scripts\run_all.ps1 -Phase experiment
.\scripts\run_all.ps1 -Phase analysis
.\scripts\run_all.ps1 -Phase finalize
```

Jika seluruh input dan cache tersedia, jalankan `.\scripts\run_all.ps1 -Phase all`. Pada Bash, gunakan `bash scripts/run_all.sh prepare|experiment|analysis|finalize|all`.

Fase `experiment` memeriksa cache embedding dan reranker lebih dahulu. Jika belum ada, runner berhenti dengan daftar file yang diperlukan. Stage 7 memakai `--resume` untuk mempertahankan seed yang sudah lengkap.

## Cache GPU privat di Kaggle

### Embedding

1. Jalankan `scripts/prepare_kaggle_embeddings.py`.
2. Buat dataset Kaggle **Private** dari `kaggle_io/embeddings/`.
3. Impor `notebooks/run_embeddings.ipynb`, pasang dataset, aktifkan GPU dan Internet.
4. Unduh keluaran dan jalankan `scripts/verify_embeddings.py`.
5. Hentikan sesi dan hapus dataset privat.

Embedding terverifikasi memakai `intfloat/multilingual-e5-base` revision `d128750597153bb5987e10b1c3493a34e5a4502a`. Notebook `run_stage6_dense.ipynb` dan `run_stage6_reranker.ipynb` memproses kandidat Tahap 6. Model reranker utama adalah `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

## Generation opsional

Tahap generation saat ini **SKIPPED** karena browser computer-use tidak tersedia. Untuk melanjutkan:

1. Jalankan `scripts/prepare_stage11_generation.py` dan `scripts/build_stage11_notebook.py`.
2. Buat dataset Kaggle **Private** berisi `kaggle_io/gen_inputs.jsonl`.
3. Impor `notebooks/run_generation.ipynb`; aktifkan GPU T4 x2 dan Internet.
4. Jalankan `TEST_MODE=True`, periksa lima QA, lalu ubah ke `False`.
5. Unduh `gen_outputs.jsonl` dan `gen_metrics.jsonl`, hentikan sesi, dan hapus dataset.
6. Jalankan `scripts/evaluate_stage11.py`.

Jangan mengklaim hasil generation sebelum kedua keluaran memiliki 1.730 baris dan evaluator lulus.

## Verifikasi

```powershell
.\.venv\Scripts\python.exe scripts\verify_stage7.py
.\.venv\Scripts\python.exe scripts\verify_stage8.py
.\.venv\Scripts\python.exe scripts\verify_stage9.py
.\.venv\Scripts\python.exe scripts\verify_stage10.py
.\.venv\Scripts\python.exe scripts\verify_stage12.py
.\.venv\Scripts\python.exe scripts\build_stage13_artifacts.py
.\.venv\Scripts\python.exe scripts\verify_stage13.py
.\.venv\Scripts\python.exe -m pytest -q
```

## Integritas eksperimen

- Split dilakukan pada tingkat dokumen.
- Test tidak digunakan untuk memilih metode atau parameter.
- Pemilihan global memakai dev seed 1–3.
- `k`, `tau`, dan lambda disetel pada dev setiap seed sebelum test dibangun.
- Semua hasil negatif tetap dilaporkan.
- Rujukan bibliografis dalam draf harus diverifikasi pengguna dari sumber asli.

## Dokumen penting

- [Laporan akhir](reports/FINAL_REPORT.md)
- [Draf Bab III](reports/DRAFT_BAB3_METODOLOGI.md)
- [Draf Bab IV](reports/DRAFT_BAB4_HASIL.md)
- [Outline paper](paper/outline.md)
- [Reproducibility](paper/reproducibility.md)
- [Keterbatasan dan etika](paper/limitations_and_ethics.md)
