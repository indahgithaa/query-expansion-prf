# Draf Bab III — Metodologi Penelitian

## 3.1 Desain penelitian

Penelitian ini menggunakan desain eksperimen komputasional untuk membandingkan metode retrieval pada sistem question answering berbasis laporan asuhan gizi klinis. Unit pemisahan data adalah dokumen pasien, bukan pasangan tanya-jawab, agar informasi pasien yang sama tidak muncul pada koleksi, data pengembangan, dan data uji dalam satu pengulangan. Eksperimen diulang sepuluh kali menggunakan seed 1–10 untuk mengukur variasi akibat pembagian dokumen.

Sistem diasumsikan telah menerima identitas pasien yang benar. Ruang pencarian dibatasi pada kumpulan chunk laporan pasien tersebut. Koleksi pasien lain hanya digunakan sebagai sumber feedback eksternal dan sumber pasangan tanya-jawab. Metode utama tidak menggunakan LLM atau fine-tuning. Model bahasa generatif hanya direncanakan sebagai evaluasi tambahan dan tidak memengaruhi pemilihan retrieval.

## 3.2 Data penelitian

Data mentah terdiri atas 911 pasangan tanya-jawab dari 120 laporan dan 13.737 chunk. Setiap QA memiliki identitas dokumen, nomor halaman, indeks chunk, pertanyaan, konteks, jawaban, diagnosis medis, asesmen antropometri, dan judul laporan. Delapan tipe pertanyaan adalah skrining, status gizi, biokimia, diagnosis PES, tujuan diet, prinsip diet, syarat diet, dan pedoman.

Nama file dinormalisasi dengan Unicode NFKC, konversi pola `#Uxxxx`, serta penghapusan karakter U+2060. Duplikasi didefinisikan melalui pasangan nama file ternormalisasi dan pertanyaan. Jika terdapat duplikasi, baris dengan ID numerik terkecil dipertahankan. Dari 911 QA, 53 baris duplikat dihapus sehingga tersisa 858 QA. Seluruh 858 QA berhasil dihubungkan ke chunk menggunakan identitas `(file, page_number, chunk_index)`. `answer_start` tidak digunakan karena hanya 79 dari 911 offset mentah yang valid.

Label utama adalah satu chunk gold ketat dari identitas metadata. Label multi-gold dibentuk sebagai analisis sensitivitas. Misalkan A adalah himpunan token unik jawaban setelah lowercase, tokenisasi, dan penghapusan stopword. Chunk C di dokumen yang sama dianggap relevan jika

$$
\frac{|A\cap \operatorname{token}(C)|}{|A|}\geq t,
$$

dengan ambang $t\in\{0{,}5;0{,}6;0{,}7\}$. Gold ketat selalu dimasukkan. QA dengan kurang dari tiga token jawaban dikeluarkan dari evaluasi multi-gold. Sebanyak 770 QA memenuhi syarat.

## 3.3 Pembagian data

Setiap seed membagi 120 dokumen menjadi 72 dokumen koleksi (60%), 24 dev (20%), dan 24 test (20%). Semua tipe pertanyaan harus muncul pada setiap bagian. Jumlah QA test per seed bervariasi antara 169 dan 177; setelah empat skenario dibentuk, total evaluasi test mencapai 6.888 observasi query-skenario.

Koleksi tanya-jawab $\mathcal{A}$ berisi pasangan pertanyaan-jawaban dari dokumen koleksi. Korpus eksternal $\mathcal{E}$ berisi seluruh chunk dari dokumen koleksi. Dokumen target $D_p$ berisi chunk milik pasien yang sedang ditanyakan. Pemeriksaan otomatis memastikan dokumen dev/test tidak muncul dalam $\mathcal{A}$ atau $\mathcal{E}$.

Empat skenario uji digunakan. TPL memakai pertanyaan asli dan koleksi lengkap. NEW memakai pertanyaan asli tetapi seluruh QA dengan tipe sama dikeluarkan dari $\mathcal{A}$. PARA memakai parafrasa dan koleksi lengkap. PNEW memakai parafrasa dan mengeluarkan tipe yang sama. Korpus $\mathcal{E}$ tetap lengkap pada semua skenario. Enam parafrasa dev dan enam parafrasa test yang berbeda disusun untuk masing-masing delapan tipe, sehingga tersedia 96 parafrasa. Pemilihan parafrasa per QA bersifat deterministik berdasarkan seed dan QID.

## 3.4 Pra-pemrosesan

Teks diubah menjadi lowercase dan ditokenisasi memakai regex `\w+`. Stopword menggunakan daftar Sastrawi ditambah kata `apa`, `bagaimana`, `yang`, `ini`, dan `pasien`. Query BM25 untuk pencarian chunk memakai versi tanpa stopword, dilambangkan $q'$. Question similarity QS-BM25 memakai pertanyaan tanpa penghapusan stopword agar bentuk kalimat tetap informatif.

BM25 menggunakan $k_1=1{,}5$ dan $b=0{,}75$. Untuk term $t$ dan dokumen $d$, skor dihitung sebagai

$$
\operatorname{BM25}(q,d)=\sum_{t\in q}\operatorname{IDF}(t)
\frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b|d|/\operatorname{avgdl})}.
$$

Implementasi mengikuti kebijakan IDF `rank_bm25`: IDF negatif diganti dengan $\epsilon$ dikali rata-rata IDF. Ranking chunk memakai urutan skor menurun dengan pemecahan seri stabil berdasarkan indeks chunk. Ranking pertanyaan memakai pemecahan seri acak deterministik berbasis seed dan identitas query.

## 3.5 Metode pembanding

B0 adalah BM25 dengan query asli. B1 adalah baseline utama BM25 dengan query setelah stopword dihapus. B1-DENSE menggunakan cosine similarity embedding `multilingual-e5-base`; B1-HYB menggabungkan B1 dan dense menggunakan reciprocal rank fusion (RRF). B1+RR menerapkan reranker pada top-20 B1.

RM3 mewakili PRF klasik dari $D_p$. Kombinasi `fb_docs` {3,5,10}, `fb_terms` {10,20,30}, dan interpolasi {0,3; 0,5; 0,7} dipilih pada dev. Pilihan akhir adalah lima dokumen feedback, 30 term, dan interpolasi 0,5.

Ablation mencakup P-EXT (hanya feedback eksternal), P-QA (hanya feedback QA), P-FUS (RRF kedua ranking), ZONE3 (aturan keras tiga zona), SOFT-SM (bobot softmax antarjawaban), serta ORACLE-W. ORACLE-W memilih secara retrospektif ranking P-QA atau P-EXT yang memberi rank gold terbaik, sehingga hanya merupakan batas atas analitis.

## 3.6 Metode SOFT

Untuk pertanyaan baru $q$, question similarity dihitung terhadap setiap pertanyaan $q_j$ di $\mathcal{A}$:

$$s_j=\operatorname{sim}(q,q_j),\qquad s=\max_j s_j.$$

Lima pertanyaan teratas dipilih; teks jawabannya menjadi feedback QA. Secara terpisah, lima chunk teratas dari $\mathcal{E}$ dicari dengan $q'$ dan menjadi feedback eksternal. Dua query berbobot dibentuk:

$$q_{QA}=r\cdot q' + \sum_{i=1}^{m}\operatorname{token}(a_i),$$

$$q_{EXT}=r\cdot q' + \sum_{i=1}^{m}\operatorname{token}(e_i),$$

dengan $m=5$ dan $r=3$. Kedua query menghasilkan skor $S_{QA}(c)$ dan $S_{EXT}(c)$ untuk setiap chunk target. Skor dinormalisasi min–max per query:

$$\widehat S(c)=\frac{S(c)-\min_{c'\in D_p}S(c')}{\max_{c'\in D_p}S(c')-\min_{c'\in D_p}S(c')}.$$

Bobot sumber QA ditentukan oleh sigmoid:

$$w(s)=\frac{1}{1+e^{-k(s-\tau)}}.$$

Skor SOFT adalah

$$S_{SOFT}(c)=w(s)\widehat S_{QA}(c)+(1-w(s))\widehat S_{EXT}(c).$$

Nilai $k$ dipilih dari {0,5; 1; 2; 4; 8}. Nilai $\tau$ dipilih dari persentil ke-5 sampai ke-95 distribusi $s$ dengan langkah lima. Keduanya dituning ulang pada dev untuk setiap seed final.

## 3.7 Komponen semantik dan reranker

Kandidat question similarity terdiri atas QS-BM25, QS-EMB, serta hybrid min–max dan RRF. Embedding menggunakan `intfloat/multilingual-e5-base` dengan awalan `query:` dan vektor ternormalisasi.

Kandidat retriever adalah R-BM25, R-DENSE, dan R-HYB. Ekspansi dense diuji melalui embedding teks penuh dan Rocchio:

$$\mathbf v=\alpha\operatorname{emb}(q)+\beta\frac{1}{m}\sum_{i=1}^{m}\operatorname{emb}(f_i),$$

dengan $\alpha,\beta\in\{0{,}25;0{,}5;0{,}75;1\}$. Hybrid memakai RRF dengan konstanta 60:

$$S_{RRF}(c)=\sum_l\frac{1}{60+\operatorname{rank}_l(c)}.$$

Reranker `BAAI/bge-reranker-v2-m3` menilai pasangan pertanyaan asli dan top-20 chunk. Skor gabungan adalah

$$S_{FINAL}(c)=\lambda\widehat S_{RR}(c)+(1-\lambda)\widehat S_{SOFT}(c),$$

dengan $\lambda\in\{0;0{,}25;0{,}5;0{,}75;1\}$. Chunk di luar top-20 mempertahankan urutan SOFT di bawah kandidat.

## 3.8 Pemilihan konfigurasi

Pemilihan komponen dilakukan berurutan pada dev seed 1–3: (1) $m$ dan $r$; (2) question similarity; (3) retriever; (4) reranker. Tujuan adalah MRR rata-rata empat skenario tertinggi dengan syarat tidak ada skenario turun lebih dari 0,01 terhadap pilihan sebelumnya. Jika tidak ada kandidat yang lolos, komponen sebelumnya dipertahankan.

Hasilnya adalah $m=5$, $r=3$, QS-BM25, R-BM25, dan reranker top-20 dengan lambda acuan 0,25. Pada run final, $k$, $\tau$, lambda B1+RR, dan lambda FINAL dipilih ulang hanya dari dev masing-masing seed. Test baru diproses setelah parameter terkunci. Konfigurasi tersimpan di `configs/final.yaml` dengan `test_used_for_selection: false`.

## 3.9 Metrik

Untuk satu gold, reciprocal rank adalah $RR=1/r$, MRR adalah rata-rata RR, dan Recall@$k$ adalah proporsi query dengan gold pada top-$k$. nDCG@10 memakai gain biner dan ideal ranking satu. Untuk multi-gold, MAP@50 dan nDCG@10 dihitung dari ranking tersimpan.

Ketahanan terhadap B1 diukur melalui win/tie/loss dan Robustness Index:

$$RI=\frac{N^+-N^-}{N},$$

dengan $N^+$ jumlah rank yang membaik dan $N^-$ jumlah rank yang memburuk. Tie masuk penyebut tetapi tidak pembilang.

## 3.10 Analisis statistik

Wilcoxon signed-rank dua sisi diterapkan pada RR berpasangan, baik per seed maupun pada gabungan observasi. Tujuh perbandingan utama dikoreksi dengan Holm. CI 95% delta MRR diperoleh melalui 10.000 paired bootstrap resample yang distratifikasi pada kelompok seed-skenario. Effect size memakai matched-pairs rank-biserial. Bukti tidak hanya bertumpu pada p-value gabungan karena QA dapat muncul pada seed berbeda dan skenario saling berkaitan; jumlah seed signifikan dan CI turut dilaporkan.

## 3.11 Reproduksibilitas dan privasi

Kode, konfigurasi, laporan agregat, tabel, dan gambar dapat dilacak Git. Data klinis, hasil per-query, model, cache, kredensial, dan folder pertukaran Kaggle diabaikan. Dataset Kaggle yang memuat teks harus Private dan dihapus setelah hasil terverifikasi. Seed, revision model, hash artefak, parameter dev, dan pemeriksaan kebocoran disimpan untuk audit.

## 3.12 Catatan rujukan

Landasan yang perlu dicantumkan mencakup BM25, Rocchio/relevance model, selective query expansion, question retrieval, RRF, mE5, cross-encoder reranking, RAG, dan Robustness Index. Detail bibliografis yang tercantum pada rancangan awal **belum diverifikasi satu per satu dari sumber asli**. Pengguna wajib memeriksa judul, penulis, tahun, DOI, dan kesesuaian klaim sebelum menjadikan draf ini naskah final.

