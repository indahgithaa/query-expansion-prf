# Log Keputusan Desain

| Tanggal | Keputusan | Alasan |
|---|---|---|
| 2026-10-15 | Menggunakan repositori aktif `C:\Users\indah\skripsi\query-expansion-prf`. | Folder ini telah disiapkan pengguna sebagai repositori baru dan hanya berisi data lokal tanpa commit. |
| 2026-10-15 | Memakai Python 3.12 dan versi dependensi yang dipin di `requirements.txt`. | Versi tersebut tersedia dan telah terbukti kompatibel pada lingkungan referensi lokal dengan versi Python yang sama. |
| 2026-10-15 | Memakai `data/chunks/*.chunks.json` sebagai sumber chunk. | Arsip `chunks.zip` tidak ditemukan, tetapi 120 berkas hasil ekstraksinya tersedia lengkap; pemrosesan tidak memerlukan arsip asli. |
| 2026-10-15 | Menjaga seluruh `data/`, `results/`, `models/`, `.env`, cache, dan `kaggle_io/` di luar Git. | Teks laporan klinis tetap dianggap sensitif meskipun sudah disamarkan. |
| 2026-10-15 | Menurunkan pin NumPy dari 2.5.4 ke 2.5.3. | Build 2.5.4 diblokir Windows Application Control pada `_pocketfft_umath`; build 2.5.3 terverifikasi dapat dimuat dengan SciPy 1.18.1 dan scikit-learn 1.9.1 di mesin yang sama. |
| 2026-10-15 | Menetapkan identitas chunk dari `(nama file ternormalisasi, page_number, chunk_index)`. | Join metadata berhasil 911/911, sedangkan 12 `context` tidak sama persis dengan teks chunk; identitas metadata lebih stabil daripada kesamaan string. |
| 2026-10-15 | Mengabaikan `answer_start` untuk seluruh pipeline. | Hanya 79/911 offset yang valid, sesuai masalah data yang sudah diantisipasi. |
| 2026-10-15 | Pada duplikasi `(file, question)`, mempertahankan ID numerik terkecil. | Kebijakan deterministik, dapat diaudit, dan tidak menggunakan performa retrieval atau bagian test; 53 baris dibuang dari 35 kelompok. |
| 2026-10-15 | Membentuk split 72/24/24 dokumen dengan shuffle deterministik per seed dan validasi kedelapan tipe. | Menjaga rasio 60/20/20, mencegah kebocoran pasien, dan menjamin semua skenario dapat dihitung di setiap bagian. |
| 2026-10-15 | Memakai himpunan token unik jawaban untuk rasio overlap multi-gold. | Implementasi langsung definisi `|A ∩ token(C)| / |A|`; gold ketat selalu dipertahankan dan 88 QA dengan kurang dari tiga token dikeluarkan. |
| 2026-10-15 | Tidak menjalankan embedding E5-base di CPU. | Benchmark 64 chunk hanya 0,620 item/detik dan memperkirakan 28.682 detik (7,97 jam), melampaui batas dua jam. |
| 2026-10-15 | Mencoba model cadangan `intfloat/multilingual-e5-small` secara lokal setelah Kaggle tidak tersedia. | Inventaris computer-use kosong dan tiga jalur browser (Chrome, peluncuran Chrome, IAB) gagal sebelum login; fallback gratis telah ditetapkan dalam protokol dan akan digunakan hanya bila estimasi CPU <2 jam. |
| 2026-10-15 | Mengestimasi embedding berdasarkan teks unik untuk pertanyaan dan jawaban. | Delapan template pertanyaan tidak perlu dihitung 858 kali; cache tetap diekspansi kembali ke urutan seluruh QA, sehingga optimasi tidak mengubah skor. |
| 2026-10-15 | Menghentikan Tahap 3 sebelum cache embedding penuh. | E5-base CPU diperkirakan 7,97 jam; tiga konfigurasi E5-small tetap 2,14–3,04 jam atau lebih, sementara runtime computer-use tidak menyediakan browser untuk Kaggle. Tahap wajib telah gagal setelah tiga pendekatan berbeda. |

