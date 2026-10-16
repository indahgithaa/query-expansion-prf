# Tahap 11 — Generation di Kaggle

## Status

**SKIPPED** pada 2026-10-16. Tahap ini bersifat opsional dan akses computer-use gagal setelah tiga percobaan berbeda sebelum halaman Kaggle dapat dikendalikan.

## Yang berhasil disiapkan

- `kaggle_io/gen_inputs.jsonl` berisi 1.730 baris: 173 QA seed 1 × dua skenario (TPL dan PARA) × lima konfigurasi (B1, RM3, SOFT, FINAL, ORACLE).
- Setiap pasangan `(id, configuration)` unik; masing-masing dari 10 kombinasi skenario-konfigurasi berisi tepat 173 baris.
- B1, RM3, SOFT, dan FINAL memakai tiga chunk teratas dari ranking beku Tahap 7. ORACLE memakai chunk gold sebagai K1, sedangkan K2 dan K3 dikosongkan agar konteks oracle tidak tercampur chunk non-gold.
- Pertanyaan yang dikirim ke generator selalu pertanyaan asli dari `data/processed/qa.jsonl`; skenario hanya menentukan ranking konteks.
- `notebooks/run_generation.ipynb` telah disiapkan dengan gate Python/GPU/internet, kandidat versi `transformers` dan `bitsandbytes` yang dipin, kuantisasi 4-bit, model utama dan fallback Qwen2.5, mode uji 5 QA, cache per `(configuration, id)`, greedy decoding maksimal 256 token, ROUGE-L, BERTScore multilingual, serta pemeriksaan kelengkapan.
- `scripts/evaluate_stage11.py` telah disiapkan untuk validasi hasil penuh, agregasi metrik, uji Wilcoxon FINAL vs B1 dan SOFT vs B1, serta pembuatan ulang laporan ketika output tersedia.

SHA-256 input privat: `880dc8ecd77794d60016d949da69d766ddcfff0eb3fbd78c3cf424472a485fa7`.

## Tiga percobaan computer-use

| Percobaan | Jalur | Hasil |
|---:|---|---|
| 1 | Inventaris permukaan dan Chrome | `apps: []`, `browsers: []`; Chrome: `Browser is not available: chrome` |
| 2 | Microsoft Edge | `Browser is not available: edge` |
| 3 | In-app browser | `Browser is not available: iab` |

Kegagalan terjadi sebelum login, pembuatan dataset, atau upload. Tidak ada teks laporan yang meninggalkan repository dan tidak ada dataset/sesi Kaggle yang perlu dihapus atau dihentikan.

## Metrik generation

ROUGE-L F1, BERTScore F1, abstain rate, serta Wilcoxon **tidak dilaporkan**, karena tidak ada jawaban model yang berhasil dihasilkan. Angka tidak diimputasi dan tidak diturunkan dari metrik retrieval.

## Cara melanjutkan bila computer-use tersedia

1. Buat dataset Kaggle **Private** dan unggah `kaggle_io/gen_inputs.jsonl`.
2. Impor `notebooks/run_generation.ipynb`, pasang dataset sebagai input, aktifkan **GPU T4 x2** dan **Internet On**.
3. Jalankan `TEST_MODE = True`; periksa lima QA. Jika wajar dan berbahasa Indonesia, ubah menjadi `False` lalu jalankan ulang seluruh notebook.
4. Unduh `gen_outputs.jsonl` dan `gen_metrics.jsonl` ke `kaggle_io/`, hentikan sesi, lalu hapus dataset privat setelah kelengkapan lokal terverifikasi.
5. Jalankan `.venv\\Scripts\\python.exe scripts\\evaluate_stage11.py`.

Versi paket pada notebook masih berstatus kandidat karena instalasi Kaggle belum dapat diuji. Jangan menyatakan hasil generation tersedia sebelum notebook penuh lolos dan kedua file keluaran berisi 1.730 baris.
