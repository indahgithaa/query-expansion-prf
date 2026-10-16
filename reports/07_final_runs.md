# Run Final 10 Pengulangan

## Protokol

Run final memakai konfigurasi komponen yang telah dikunci di `configs/final.yaml`: `m=5`, `r=3`, QS-BM25, R-BM25, serta reranker `BAAI/bge-reranker-v2-m3` pada top-20. Untuk setiap seed 1–10, urutan proses dipaksa sebagai berikut:

1. Bangun fitur partisi dev.
2. Pilih `k` dan `tau` dari grid dev seed tersebut.
3. Pilih lambda B1+RR dan FINAL pada dev dengan kendala tidak ada skenario turun lebih dari 0,01 terhadap lambda nol.
4. Setelah seluruh parameter terkunci, baru bangun fitur test dan evaluasi 16 konfigurasi.

Parameter RM3 (`fb_docs=5`, `fb_terms=30`, interpolasi 0,5), temperatur SOFT-SM (1,0), dan persentil ZONE3 (33/67) tetap mengikuti keputusan Tahap 5. Nilai batas ZONE3 dihitung dari distribusi skor dev setiap seed tanpa memakai label test.

## Parameter terpilih per seed

| Seed | Query dev | Query test | k | tau | lambda B1+RR | lambda FINAL |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 696 | 692 | 0,5 | 1,629308 | 0,75 | 0,25 |
| 2 | 684 | 696 | 1,0 | 3,119274 | 0,75 | 0,25 |
| 3 | 672 | 708 | 2,0 | 3,475055 | 1,00 | 0,00 |
| 4 | 688 | 688 | 0,5 | 3,120461 | 1,00 | 0,25 |
| 5 | 680 | 692 | 0,5 | 3,488131 | 0,75 | 0,00 |
| 6 | 688 | 676 | 1,0 | 3,014089 | 0,75 | 0,25 |
| 7 | 692 | 680 | 1,0 | 3,577024 | 0,75 | 0,25 |
| 8 | 692 | 680 | 0,5 | 2,746055 | 0,75 | 0,00 |
| 9 | 680 | 680 | 1,0 | 3,747420 | 0,75 | 0,00 |
| 10 | 684 | 696 | 1,0 | 2,869448 | 1,00 | 0,25 |

Ringkasan frekuensi:

- `k=0,5` pada 4 seed, `k=1` pada 5 seed, dan `k=2` pada 1 seed.
- Lambda B1+RR 0,75 pada 7 seed dan 1,00 pada 3 seed.
- Lambda FINAL 0,25 pada 6 seed dan 0,00 pada 4 seed. Nilai nol dipertahankan ketika reranker tidak lolos atau tidak memperbaiki pilihan dev sesuai aturan.

## Konfigurasi yang dievaluasi

Seluruh konfigurasi wajib dijalankan: B0, B1, B1-DENSE, B1-HYB, B1+RR, RM3, P-EXT, P-QA, P-FUS, ZONE3, SOFT, SOFT-SM, SOFT-QS*, SOFT-QS*-R*, FINAL, dan ORACLE-W.

Karena QS-BM25 dan R-BM25 terpilih pada Tahap 6, SOFT, SOFT-QS*, dan SOFT-QS*-R* identik secara konstruksi. Ketiganya tetap disimpan sebagai label terpisah agar tabel dekomposisi Tahap 9 lengkap dan dapat diaudit.

## Artefak privat

- `results/runs/seed_XX_rankings.jsonl.gz`: satu baris per kombinasi query–metode, berisi ranking maksimal 50 `chunk_id`.
- `results/runs/seed_XX_queries.csv.gz`: rank gold dan variabel audit per query untuk seluruh metode.
- `results/params/seed_XX.yaml`: parameter yang dipilih dari dev dan hash artefak test.
- `results/params/grids/`: grid sigmoid dan reranker per seed.
- `results/stage7/audit.json` dan `results/stage7/verification.json`: audit gabungan serta hasil verifier independen.

Jika laporan pasien memiliki kurang dari 50 chunk, seluruh chunk disimpan. Empat dokumen test masuk kondisi ini; verifier membandingkan panjang aktual terhadap `min(50, jumlah chunk pasien)`.

## Verifikasi

- 10/10 seed selesai.
- 6.888 observasi query–skenario test dievaluasi.
- 110.208 baris ranking tersimpan (`6.888 × 16 metode`).
- Dua puluh hash artefak ranking/query cocok dengan file parameter.
- Setiap query memiliki tepat 16 metode; tidak ada `chunk_id` duplikat dalam ranking.
- Seluruh chunk hasil retrieval berasal dari dokumen pasien yang benar.
- Posisi gold pada ranking cocok dengan rank yang disimpan di CSV.
- Audit menyatakan `selection_partition: dev`, `evaluation_partition: test`, dan `test_used_for_tuning: false` pada semua seed.
- Total waktu proses sepuluh seed sekitar 808,14 detik (13,47 menit).

Tahap ini hanya menghasilkan dan memverifikasi run. Agregasi metrik, uji statistik, dan interpretasi hasil test ditunda ke Tahap 8.
