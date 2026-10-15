# Log Keputusan Desain

| Tanggal | Keputusan | Alasan |
|---|---|---|
| 2026-10-15 | Menggunakan repositori aktif `C:\Users\indah\skripsi\query-expansion-prf`. | Folder ini telah disiapkan pengguna sebagai repositori baru dan hanya berisi data lokal tanpa commit. |
| 2026-10-15 | Memakai Python 3.12 dan versi dependensi yang dipin di `requirements.txt`. | Versi tersebut tersedia dan telah terbukti kompatibel pada lingkungan referensi lokal dengan versi Python yang sama. |
| 2026-10-15 | Memakai `data/chunks/*.chunks.json` sebagai sumber chunk. | Arsip `chunks.zip` tidak ditemukan, tetapi 120 berkas hasil ekstraksinya tersedia lengkap; pemrosesan tidak memerlukan arsip asli. |
| 2026-10-15 | Menjaga seluruh `data/`, `results/`, `models/`, `.env`, cache, dan `kaggle_io/` di luar Git. | Teks laporan klinis tetap dianggap sensitif meskipun sudah disamarkan. |
| 2026-10-15 | Menurunkan pin NumPy dari 2.5.4 ke 2.5.3. | Build 2.5.4 diblokir Windows Application Control pada `_pocketfft_umath`; build 2.5.3 terverifikasi dapat dimuat dengan SciPy 1.18.1 dan scikit-learn 1.9.1 di mesin yang sama. |

