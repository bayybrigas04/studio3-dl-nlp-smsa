# 📂 Hasil Eksperimen — Fine-Tuning Transformer untuk Klasifikasi Sentimen SmSA

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow.svg)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-SmSA%20(IndoNLU)-green.svg)](https://github.com/indonlp/indonlu)
[![GPU](https://img.shields.io/badge/GPU-Tesla%20T4-76B900.svg)](#%EF%B8%8F-konfigurasi-eksperimen)
[![Runs](https://img.shields.io/badge/Runs-27%20selesai-success.svg)](#-struktur-folder)

Folder ini memuat **seluruh keluaran eksperimen** dari notebook `Studio_3_RA_DL_NLP.ipynb`
(Studio Proyek 3 — Pembelajaran Mendalam IF25-40401, Institut Teknologi Sumatera, Kelompok 2):
log mentah 27 run, rekap agregat, benchmark inferensi, prediksi uji, baseline pembanding, dan gambar.

> **Kode konfigurasi `5579c1`** · sesi `20261120-081311` · GPU **Tesla T4 (14,6 GB VRAM)** · PyTorch `2.14.1+cu130`
> Seluruh angka pada dokumen ini bersumber dari berkas di folder ini dan output sel notebook.

**Navigasi:** [Ringkasan](#-ringkasan-hasil) · [Struktur](#-struktur-folder) · [Konfigurasi](#%EF%B8%8F-konfigurasi-eksperimen) · [Hasil](#-hasil-eksperimen) · [Baseline](#-pembanding-baseline) · [Hipotesis](#-verifikasi-hipotesis) · [Error Analysis](#-analisis-kesalahan) · [Gambar](#%EF%B8%8F-gambar) · [Kamus Kolom](#-kamus-kolom) · [Catatan](#%EF%B8%8F-catatan-dan-keterbatasan)

---

## 🏆 Ringkasan Hasil

| | Temuan |
|:---|:---|
| 🥇 **Kombinasi terbaik** | **IndoBERT + Full Fine-Tuning**: F1 makro **0,8654 ± 0,0337**, akurasi **0,9047 ± 0,0163**, F1 `neutral` **0,7306 ± 0,0872** |
| ⚖️ **Trade-off terbaik** | **IndoBERT + Partial Freezing**: F1 makro **0,8499 ± 0,0316**; selisih terhadap Full (−0,0155) **belum bermakna** (≤ SD antar-seed), hemat waktu latih **43,5%** dan memori GPU **48,7%** |
| ⚡ **Inferensi tercepat** | **DistilBERT**: **0,742 ms/kalimat** (1.347,7 kalimat/detik), tetapi F1 makro turun **5,90 poin** dari BERT |
| 🪫 **Paling hemat, paling lemah** | **Linear Probing**: hemat waktu latih 74,3%–77,5%, namun pada BERT multibahasa F1 makro hanya **0,3755** (tebakan acak: 0,3247) |
| 🎯 **Kelas tersulit** | `neutral` (±10% data latih). F1 `neutral` terbaik hanya **0,7306**, dengan variasi antar-seed terbesar (SD 0,0872) |

---

## 📁 Struktur Folder

```text
output_studio3/
├── README.md                                # dokumen ini
├── hasil_smsa_5579c1.csv                    # log mentah 27 run (1 baris per run)
├── rekap_agregat_smsa_5579c1.csv            # rata-rata ± SD per backbone × strategi (9 baris)
├── benchmark_inferensi_smsa_5579c1.csv      # latensi & throughput terkontrol dalam satu sesi GPU
├── baseline.csv                             # TF-IDF + LogReg, tebakan acak, kelas mayoritas
├── preds_5579c1/                            # 27 berkas prediksi uji terkompresi
│   ├── IndoBERT_full_seed42.npz             # berisi larik gold & pred (untuk Confusion Matrix)
│   └── ...                                  # {Backbone}_{strategi}_seed{seed}.npz
└── figures/
    ├── 01_duplikasi_train.png
    ├── 02_perbandingan_backbone_strategi.png
    ├── 03_kurva_pelatihan.png
    ├── 04_cm_IndoBERT_full_vs_probing.png
    └── 05_cm_backbone_{full,partial,probing}.png
```

> 💡 **Arti `5579c1`:** kode hash konfigurasi (hiperparameter + daftar backbone + strategi). Hasil dari konfigurasi
> berbeda otomatis mendapat kode berbeda sehingga tidak tercampur. **Jangan mengganti nama berkas**, karena
> notebook mencari berkas berdasarkan pola ini.

---

## ⚙️ Konfigurasi Eksperimen

Matriks **27 run** = 3 backbone × 3 strategi × 3 seed.

| Komponen | Nilai |
|:---|:---|
| **Backbone** | `indobenchmark/indobert-base-p1` (124.443.651 param, vocab 30.521) · `bert-base-multilingual-cased` (177.855.747 param, vocab 119.547) · `distilbert-base-multilingual-cased` (135.326.979 param, vocab 119.547) |
| **Strategi** | **Full** (100% parameter) · **Partial** (embedding + separuh lapis awal beku: 6/12 pada IndoBERT & BERT, 3/6 pada DistilBERT) · **Probing** (seluruh `base_model` beku; hanya head dilatih) |
| **Seed** | `42, 43, 44` |
| **Learning rate** | Full `3e-5` · Partial `3e-5` · Probing `2e-3` |
| **Batch / max length** | 32 (latih), 64 (evaluasi) / 128 token (*dynamic padding*) |
| **Epoch** | 3; epoch terbaik dipilih dari **F1 makro validasi** tertinggi |
| **Optimizer** | AdamW (`weight_decay = 0.01`), warmup 10% + linear decay, `max_grad_norm = 1.0`, FP16 |
| **Metrik utama** | F1 makro (kelas tidak seimbang), akurasi, F1 per kelas (khususnya `neutral`), waktu latih, latensi inferensi, memori puncak |
| **Pelaporan** | Rata-rata ± simpangan baku sampel (`ddof = 1`) antar 3 seed; data uji disentuh **sekali** per run |

### Pembagian Data

| Himpunan | Sumber | Awal | Dipakai | Peran |
|:---|:---|:---:|:---:|:---|
| **Latih** | `train_preprocess.tsv` | 11.000 | **5.000** (dari 10.933 bersih) | subset *stratified*, `subset_seed = 42`, identik di seluruh run |
| **Validasi** | `valid_preprocess.tsv` | 1.260 | **1.246** | memilih epoch terbaik dan mengukur latensi inferensi |
| **Uji** | `test_preprocess.tsv` (label asli) | 500 | **500** | evaluasi akhir, dipakai utuh |

**Ringkasan pembersihan:** train memiliki 58 kelompok teks duplikat (125 baris, **0** konflik label) → dibuang 67 salinan
→ **10.933** baris. Validation dibuang **14** baris yang bocor dari train. Audit pada uji: **0** tumpang tindih
dengan train maupun validasi, dan **0** duplikat internal.

| Kelas | Subset latih (5.000) | Validasi (1.246) | Uji (500) |
|:---|:---:|:---:|:---:|
| positive | 2.919 (58,4%) | 728 (58,4%) | 208 (41,6%) |
| negative | 1.560 (31,2%) | 389 (31,2%) | 204 (40,8%) |
| neutral | 521 (10,4%) | 129 (10,4%) | 88 (17,6%) |

> ⚠️ Sebaran kelas **uji berbeda** dari data latih (`neutral` 17,6% vs 10,4%), sehingga skor uji tidak boleh dibaca
> sebagai estimasi sebaran latih.

**Panjang token** (subset latih, `max_length = 128`):

| Tokenizer | Vocab | Rata-rata token | Persentil-95 | Maksimum | Terpotong (>128) |
|:---|:---:|:---:|:---:|:---:|:---:|
| BERT / DistilBERT (multibahasa) | 119.547 | 50,0 | 104 | 176 | 0,18% |
| IndoBERT | 30.521 | 37,8 | 79 | 132 | 0,02% |

---

## 📊 Hasil Eksperimen

Rata-rata ± simpangan baku 3 seed pada **500 kalimat uji resmi**. Sumber: `rekap_agregat_smsa_5579c1.csv`.

### Kinerja dan Efisiensi (9 Kombinasi)

| Backbone | Strategi | F1 Makro | F1 Neutral | Akurasi | Waktu Latih (s) | Memori Puncak (MB) | Inferensi (ms/kal) | Parameter Dilatih |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **IndoBERT** | **Full** | **0,8654 ± 0,0337** | **0,7306 ± 0,0872** | **0,9047 ± 0,0163** | 75,8 ± 0,4 | 3.738 ± 5 | 1,188 ± 0,018 | 124.443.651 (100,00%) |
| IndoBERT | Partial | 0,8499 ± 0,0316 | 0,6996 ± 0,0783 | 0,8940 ± 0,0156 | 42,8 ± 0,3 | 1.918 ± 4 | 1,189 ± 0,021 | 43.120.131 (34,65%) |
| IndoBERT | Probing | 0,7197 ± 0,0161 | 0,5235 ± 0,0202 | 0,7807 ± 0,0130 | 19,4 ± 0,1 | 650 ± 0 | 1,186 ± 0,020 | 2.307 (<0,01%) |
| BERT (mBERT) | Full | 0,7370 ± 0,0233 | 0,5537 ± 0,0275 | 0,7927 ± 0,0221 | 99,9 ± 0,8 | 4.577 ± 15 | 1,548 ± 0,036 | 177.855.747 (100,00%) |
| BERT (mBERT) | Partial | 0,7046 ± 0,0098 | 0,5844 ± 0,0368 | 0,7420 ± 0,0020 | 52,3 ± 0,1 | 2.138 ± 7 | 1,540 ± 0,002 | 43.120.131 (24,24%) |
| BERT (mBERT) | Probing | 0,3755 ± 0,0124 | 0,0931 ± 0,0115 | 0,4867 ± 0,0110 | 25,0 ± 0,1 | 877 ± 0 | 1,523 ± 0,011 | 2.307 (<0,01%) |
| DistilBERT | Full | 0,6780 ± 0,0166 | 0,4945 ± 0,0235 | 0,7367 ± 0,0151 | 58,1 ± 0,1 | 3.018 ± 6 | 0,763 ± 0,002 | 135.326.979 (100,00%) |
| DistilBERT | Partial | 0,6145 ± 0,0171 | 0,4713 ± 0,0416 | 0,6653 ± 0,0081 | 25,6 ± 0,1 | 1.275 ± 3 | 0,765 ± 0,008 | 21.856.515 (16,15%) |
| DistilBERT | Probing | 0,5187 ± 0,0069 | 0,3443 ± 0,0367 | 0,5820 ± 0,0020 | 13,1 ± 0,1 | 710 ± 0 | 0,763 ± 0,009 | 592.899 (0,44%) |

### F1 per Kelas

| Backbone | Strategi | F1 Positive | F1 Neutral | F1 Negative |
|:---|:---|:---:|:---:|:---:|
| IndoBERT | Full | 0,9216 ± 0,0195 | 0,7306 ± 0,0872 | 0,9440 ± 0,0057 |
| IndoBERT | Partial | 0,9160 ± 0,0132 | 0,6996 ± 0,0783 | 0,9339 ± 0,0038 |
| IndoBERT | Probing | 0,8032 ± 0,0185 | 0,5235 ± 0,0202 | 0,8324 ± 0,0113 |
| BERT (mBERT) | Full | 0,8229 ± 0,0270 | 0,5537 ± 0,0275 | 0,8344 ± 0,0216 |
| BERT (mBERT) | Partial | 0,7436 ± 0,0027 | 0,5844 ± 0,0368 | 0,7859 ± 0,0047 |
| BERT (mBERT) | Probing | 0,5729 ± 0,0035 | 0,0931 ± 0,0115 | 0,4605 ± 0,0284 |
| DistilBERT | Full | 0,7430 ± 0,0175 | 0,4945 ± 0,0235 | 0,7966 ± 0,0181 |
| DistilBERT | Partial | 0,6340 ± 0,0069 | 0,4713 ± 0,0416 | 0,7381 ± 0,0036 |
| DistilBERT | Probing | 0,5421 ± 0,0136 | 0,3443 ± 0,0367 | 0,6697 ± 0,0040 |

### Trade-off Strategi terhadap Full Fine-Tuning

Δ F1 dianggap **bermakna** hanya bila lebih besar dari SD antar-seed yang terbesar di antara kedua strategi (patokan kasar, n = 3).

| Backbone | Strategi | Δ F1 Makro | Bermakna? | Hemat Waktu Latih | Hemat Memori | Param Dilatih (% Full) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| IndoBERT | Partial | -0,0155 | Belum (≤ SD) | 43,5% | 48,7% | 34,65% |
| IndoBERT | Probing | -0,1457 | Ya | 74,3% | 82,6% | ≈0% |
| BERT (mBERT) | Partial | -0,0324 | Ya | 47,7% | 53,3% | 24,24% |
| BERT (mBERT) | Probing | -0,3615 | Ya | 75,0% | 80,8% | ≈0% |
| DistilBERT | Partial | -0,0635 | Ya | 56,0% | 57,8% | 16,15% |
| DistilBERT | Probing | -0,1593 | Ya | 77,5% | 76,5% | 0,44% |

### Hasil per Seed (F1 Makro Uji, dengan Epoch Terpilih)

| Backbone | Strategi | Seed 42 | Seed 43 | Seed 44 |
|:---|:---|:---:|:---:|:---:|
| IndoBERT | Full | 0,8803 (ep. 3) | 0,8268 (ep. 1) | 0,8891 (ep. 3) |
| IndoBERT | Partial | 0,8678 (ep. 2) | 0,8134 (ep. 1) | 0,8684 (ep. 3) |
| IndoBERT | Probing | 0,7152 (ep. 3) | 0,7062 (ep. 3) | 0,7376 (ep. 3) |
| BERT (mBERT) | Full | 0,7185 (ep. 3) | 0,7294 (ep. 3) | 0,7631 (ep. 3) |
| BERT (mBERT) | Partial | 0,7140 (ep. 3) | 0,6944 (ep. 3) | 0,7055 (ep. 3) |
| BERT (mBERT) | Probing | 0,3614 (ep. 2) | 0,3848 (ep. 3) | 0,3803 (ep. 3) |
| DistilBERT | Full | 0,6668 (ep. 3) | 0,6702 (ep. 3) | 0,6971 (ep. 3) |
| DistilBERT | Partial | 0,5949 (ep. 3) | 0,6222 (ep. 3) | 0,6264 (ep. 3) |
| DistilBERT | Probing | 0,5107 (ep. 3) | 0,5231 (ep. 3) | 0,5223 (ep. 3) |

> 🔎 **IndoBERT seed 43** memilih **epoch 1** pada strategi Full dan Partial; F1 `neutral`-nya (0,6308 pada Full)
> lebih rendah dari seed 42 (0,7692) dan seed 44 (0,7919). Inilah sumber utama SD IndoBERT yang besar pada kelas `neutral`.

### ⚡ Benchmark Inferensi Terkontrol (Satu Sesi, 5 Putaran Bergantian)

Sumber: `benchmark_inferensi_smsa_5579c1.csv`. Ketiga backbone diukur bergantian pada GPU dan sesi yang sama.
Latensi tidak bergantung pada strategi fine-tuning karena arsitektur saat inferensi identik.

| Backbone | Median (ms/kal) | Mean ± Std (ms/kal) | Throughput (kal/detik) | Parameter Total |
|:---|:---:|:---:|:---:|:---:|
| BERT (mBERT) | 1,480 | 1,478 ± 0,004 | 675,8 | 177.855.747 |
| DistilBERT | 0,742 | 0,742 ± 0,003 | 1.347,7 | 135.326.979 |
| IndoBERT | 1,158 | 1,159 ± 0,004 | 863,2 | 124.443.651 |

### Perbandingan Backbone (Full Fine-Tuning, Relatif terhadap BERT)

| Backbone | Δ F1 Makro | Parameter | Waktu Latih | Inferensi (run) | Memori Puncak |
|:---|:---:|:---:|:---:|:---:|:---:|
| **IndoBERT** | **+12,84 poin** | 70,0% (−30,0%) | 75,8% (−24,2%) | 76,8% (−23,2%) | 81,7% (−18,3%) |
| DistilBERT | −5,90 poin | 76,1% (−23,9%) | 58,1% (−41,9%) | 49,3% (−50,7%) | 65,9% (−34,1%) |

> Tokenizer IndoBERT menghasilkan urutan lebih pendek (rata-rata 37,8 vs 50,0 token), sehingga komputasi *self-attention* lebih ringan.

---

## 🎯 Pembanding (Baseline)

Sumber: `baseline.csv`, dievaluasi pada 500 kalimat uji yang sama dan dilatih tanpa GPU.

| Pembanding | Data Latih | Akurasi | F1 Makro | Waktu Latih (s) |
|:---|:---:|:---:|:---:|:---:|
| **TF-IDF + Logistic Regression (train bersih penuh)** | 10.933 | 0,8040 | **0,7423** | 5,4 |
| **TF-IDF + Logistic Regression (subset Transformer)** | 5.000 | 0,7500 | **0,6664** | 4,8 |
| Tebakan acak (*stratified*) | 10.933 | 0,3900 | 0,3247 | – |
| Selalu kelas mayoritas (`positive`) | 10.933 | 0,4160 | 0,1959 | – |

**Pembacaan:**
- Hanya **2 dari 9** kombinasi Transformer yang melampaui baseline data penuh (0,7423): **IndoBERT Full** (0,8654) dan **IndoBERT Partial** (0,8499).
- Terhadap baseline yang adil (subset 5.000 yang sama, 0,6664), **6 dari 9** kombinasi melampauinya. Yang tidak: BERT Probing, DistilBERT Partial, dan DistilBERT Probing.
- Pemeriksaan kewajaran di notebook (Cell 29) menandai di bawah baseline data penuh: BERT Full & Partial, seluruh DistilBERT, dan IndoBERT Probing; BERT Probing juga mendekati tebakan acak.

---

## 🔬 Verifikasi Hipotesis

Selisih dianggap bermakna hanya bila > simpangan baku antar-seed (patokan kasar, bukan uji statistik formal).

| Hipotesis | Hasil | Status |
|:---|:---|:---:|
| **H1.** IndoBERT memiliki F1 makro tertinggi pada Full Fine-Tuning | IndoBERT **0,8654 ± 0,0337** vs BERT **0,7370 ± 0,0233** (+0,1284) dan DistilBERT **0,6780 ± 0,0166** (+0,1874); keduanya jauh melampaui SD antar-seed | ✅ **Terbukti** |
| **H2.** DistilBERT lebih efisien dengan penurunan F1 makro ≤ 3 poin dibanding BERT | Hemat parameter **23,9%**, waktu latih **41,9%**, inferensi **50,7%**, tetapi F1 makro turun **5,90 poin** (batas 3,0; SD antar-seed ≈ 2,33 poin) | ❌ **Tidak terbukti** |
| **H3.** Urutan F1: Full ≥ Partial > Probing; waktu & memori: Full > Partial > Probing | Urutan F1 sesuai pada ketiga backbone. Partial vs Full **bermakna** pada BERT (−0,0324) dan DistilBERT (−0,0635), **belum bermakna** pada IndoBERT (−0,0155). Partial vs Probing bermakna pada ketiganya | ✅ **Terbukti** (IndoBERT: Partial setara Full secara praktis) |

---

## 🧪 Analisis Kesalahan

**Model terbaik menurut F1 validasi: IndoBERT Full, seed 42** — akurasi **0,9120**, F1 makro **0,8803**, **44 kesalahan dari 500** kalimat uji (8,8%).

| Kelas | Precision | Recall | F1 | Support |
|:---|:---:|:---:|:---:|:---:|
| positive | 0,9083 | 0,9519 | 0,9296 | 208 |
| neutral | 1,0000 | 0,6250 | 0,7692 | 88 |
| negative | 0,8943 | 0,9951 | 0,9420 | 204 |
| **macro avg** | 0,9342 | 0,8573 | **0,8803** | 500 |

**Confusion matrix** (baris = label asli, kolom = prediksi):

| IndoBERT Full (seed 42) | pred positive | pred neutral | pred negative |
|:---|:---:|:---:|:---:|
| **positive** | 198 | 0 | 10 |
| **neutral** | 19 | 55 | 14 |
| **negative** | 1 | 0 | 203 |

| IndoBERT Probing (seed 42) | pred positive | pred neutral | pred negative |
|:---|:---:|:---:|:---:|
| **positive** | 156 | 4 | 48 |
| **neutral** | 23 | 32 | 33 |
| **negative** | 1 | 1 | 202 |

- **33 dari 44 kesalahan (75%)** berasal dari kalimat berlabel `neutral`: recall `neutral` hanya 0,6250 pada Full dan 0,3636 pada Probing.
- Model Full **tidak pernah salah memprediksi `neutral`** (precision 1,0000); ia justru cenderung *menghindari* kelas ini dan memilih `positive`/`negative`.
- Linear Probing menggeser banyak kalimat `positive` (48) dan `neutral` (33) ke `negative`.
- Contoh kesalahan di notebook (Cell 34) memuat kalimat deskriptif/informatif (mis. deskripsi produk) yang diprediksi `positive`, serta kalimat bernada sarkastik atau sentimen campuran.

---

## 🖼️ Gambar

| Berkas | Isi |
|:---|:---|
| `01_duplikasi_train.png` | Sebaran duplikat data train: 52 teks muncul 2×, 3 muncul 3×, 3 muncul 4× (seluruhnya duplikat murni) |
| `02_perbandingan_backbone_strategi.png` | F1 makro, F1 `neutral`, waktu latih, dan memori puncak untuk 9 kombinasi |
| `03_kurva_pelatihan.png` | Train loss dan validation F1 makro per epoch |
| `04_cm_IndoBERT_full_vs_probing.png` | Confusion matrix model terbaik vs probing |
| `05_cm_backbone_{full,partial,probing}.png` | Confusion matrix 3 backbone per strategi (rata-rata 3 seed) |

![Perbandingan backbone dan strategi](figures/02_perbandingan_backbone_strategi.png)

![Kurva pelatihan](figures/03_kurva_pelatihan.png)

![Confusion matrix IndoBERT](figures/04_cm_IndoBERT_full_vs_probing.png)

---

## 📖 Kamus Kolom

### `hasil_smsa_5579c1.csv` (1 baris per run)

| Kolom | Arti |
|:---|:---|
| `dataset`, `kode`, `mode` | Nama dataset, kode konfigurasi, mode eksekusi (`penuh`) |
| `sesi`, `gpu` | ID sesi dan jenis GPU (waktu hanya sah dibandingkan dalam sesi yang sama) |
| `backbone`, `model_id`, `strategi`, `seed`, `lr` | Identitas run dan learning rate |
| `n_train`, `epochs`, `epoch_terbaik` | Ukuran data latih, jumlah epoch, epoch terpilih lewat validasi |
| `test_acc`, `test_f1` | Akurasi dan F1 makro uji |
| `test_f1_pos`, `test_f1_neu`, `test_f1_neg` | F1 per kelas pada data uji |
| `val_acc`, `val_f1` | Akurasi dan F1 makro validasi pada epoch terpilih |
| `val_f1_per_epoch`, `loss_per_epoch` | Riwayat per epoch, dipisah `;` |
| `waktu_latih_s` | Waktu latih murni (detik), evaluasi tidak dihitung |
| `infer_ms` | Latensi inferensi (ms/kalimat), median 3 pass pada data validasi |
| `params_total`, `params_terlatih` | Jumlah parameter total dan yang dilatih |
| `mem_puncak_mb` | Memori GPU puncak (MB) |
| `selesai` | Waktu run selesai |

### `rekap_agregat_smsa_5579c1.csv`
Kolom `*_mean` dan `*_std` adalah rata-rata dan simpangan baku sampel (`ddof = 1`) antar seed. `n_seed` adalah jumlah seed
yang dihitung dan `n_seed_waktu` jumlah seed dari sesi acuan yang dipakai untuk waktu latih dan inferensi.

### `preds_5579c1/*.npz`
Dua larik per berkas: `gold` (label asli uji, urutan sama dengan `test_preprocess.tsv`) dan `pred` (prediksi pada epoch terbaik).
Kode label: `0 = positive`, `1 = neutral`, `2 = negative`.

---

## 💻 Cara Membaca Ulang Hasil

```python
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

rekap = pd.read_csv("rekap_agregat_smsa_5579c1.csv")
hasil = pd.read_csv("hasil_smsa_5579c1.csv")

d = np.load("preds_5579c1/IndoBERT_full_seed42.npz")
labels = ["positive", "neutral", "negative"]
print(classification_report(d["gold"], d["pred"], target_names=labels, digits=4))
print(confusion_matrix(d["gold"], d["pred"], labels=[0, 1, 2]))
```

Dengan berkas `.npz`, confusion matrix dan analisis kesalahan dapat dibuat ulang **tanpa melatih model lagi**.

---

## ⚠️ Catatan dan Keterbatasan

- **Skala eksperimen:** subset latih 5.000 dari 10.933 data bersih (batas studi kasus 3.000–6.000) dan hanya **3 seed**.
- **Uji hanya 500 kalimat** dengan sebaran kelas berbeda dari data latih; selisih kecil antar model, terutama pada F1 `neutral`, perlu dibaca hati-hati.
- **Seleksi epoch** hanya dari 3 epoch, tanpa penyetelan hiperparameter terpisah per backbone.
- **"Belum bermakna" bukan "setara":** selisih ≤ SD antar-seed berarti belum terbukti berbeda dengan 3 seed, bukan bukti kesetaraan statistik.
- **Parameter probing berbeda antar backbone:** BERT/IndoBERT hanya melatih `classifier` (2.307 parameter), sedangkan DistilBERT juga melatih `pre_classifier` (592.899 parameter).
- **Waktu dan memori** bergantung pada GPU dan kondisi sesi (Tesla T4, sesi `20261120-081311`); jangan dibandingkan langsung dengan mesin lain.
- **BERT Probing (0,3755)** hampir setara tebakan acak, sehingga hasil probing pada backbone ini perlu ditafsirkan hati-hati. Penyebabnya belum diuji (dugaan: representasi fitur beku BERT multibahasa kurang sesuai untuk tugas ini), jadi jangan digeneralisasi sebagai kesimpulan umum tentang probing.

---

## 📚 Sumber

- **Dataset:** SmSA (`smsa_doc-sentiment-prosa`) dari [IndoNLU](https://github.com/indonlp/indonlu).
- **Model:** `indobenchmark/indobert-base-p1`, `bert-base-multilingual-cased`, `distilbert-base-multilingual-cased` melalui Hugging Face Transformers.
- **Notebook:** `Studio_3_RA_DL_NLP.ipynb` (Kelompok 2 — Bayu Brigas Novaldi, Martino Kelvin, Rifka Priseilla).
