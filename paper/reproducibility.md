# Reproducibility

## Lingkungan

- Windows 11 dan Python 3.12 untuk pipeline lokal.
- Dependensi dipin di `requirements.txt`.
- Embedding: `intfloat/multilingual-e5-base`, revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
- Reranker: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

Instalasi:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Input privat yang diperlukan

- `data/dataset/dataset.json`
- `data/chunks/*.chunks.json`

Cache GPU yang dapat dibangun ulang tetapi tidak didistribusikan:

- `models/embeddings/`
- `models/reranker/stage6_scores.json`
- `models/reranker/stage6_manifest.json`

Semua jalur tersebut diabaikan Git. Jangan mengganti input dengan data lain tanpa mencatat provenance dan menjalankan ulang seluruh seleksi.

## Eksekusi

Gunakan `scripts/run_all.ps1` pada Windows atau `scripts/run_all.sh` pada Bash. Runner menerima fase `prepare`, `experiment`, `analysis`, `finalize`, atau `all`. Fase eksperimen mensyaratkan cache embedding/reranker. Instruksi notebook untuk membangun cache terdapat di README dan notebook `notebooks/`.

Contoh:

```powershell
.\scripts\run_all.ps1 -Phase analysis
.\scripts\run_all.ps1 -Phase finalize
```

Runner tidak menjalankan generation secara otomatis karena membutuhkan dataset Kaggle Private, pemeriksaan lima QA, download, penghentian sesi, dan penghapusan dataset.

## Pencegahan kebocoran

- Split dilakukan pada tingkat dokumen.
- $\mathcal A$ dan $\mathcal E$ hanya dibangun dari dokumen collection.
- Pemilihan global memakai dev seed 1–3.
- Parameter per-seed dipilih pada dev sebelum test dibangun.
- `configs/final.yaml` dikunci sebelum run test.
- Verifier memeriksa `test_used_for_selection=false` dan hash keluaran.

## Artefak hasil

- `results/runs/`: ranking top-50 dan rank gold per seed.
- `results/params/`: parameter dev per seed.
- `results/stage8/`: metrik dan statistik.
- `results/stage9/`: analisis komponen.
- `results/stage10/`: analisis tipe, bobot, oracle gap, dan error.
- `results/stage12/manual_trace.json`: perhitungan manual mesin-baca.

Folder hasil tidak dilacak karena dapat memuat ID atau turunan sensitif. Laporan publik merujuk ke nama file sumber bukti tanpa menyertakan isi privat.

## Verifikasi akhir

```powershell
.\.venv\Scripts\python.exe scripts\verify_embeddings.py
.\.venv\Scripts\python.exe scripts\verify_stage7.py
.\.venv\Scripts\python.exe scripts\verify_stage8.py
.\.venv\Scripts\python.exe scripts\verify_stage9.py
.\.venv\Scripts\python.exe scripts\verify_stage10.py
.\.venv\Scripts\python.exe scripts\verify_stage12.py
.\.venv\Scripts\python.exe scripts\build_stage13_artifacts.py
.\.venv\Scripts\python.exe scripts\verify_stage13.py
.\.venv\Scripts\python.exe -m pytest -q
```

Hash tabel dan gambar paper dicatat di `paper/manifest.json`. Untuk rilis ilmiah, tambahkan commit hash Git, versi OS/driver GPU, dan tanggal eksekusi.

