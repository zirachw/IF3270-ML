#!/usr/bin/env python3
"""
Generate Prak-4/13523002_13523004_K1_Deadline1.ipynb
IF3270 Pembelajaran Mesin | Praktikum 4 | Recurrent Neural Network

Run from Prak-4/ directory:
    python build_prak4.py
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "13523002_13523004_K1_Deadline1.ipynb"

HR = "---"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def section(title: str, anchor: str) -> dict:
    return md(f"{HR}\n\n# {title} <a name=\"{anchor}\"></a>\n\n{HR}")


cells = []

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
cells.append(md(
    "---\n\n"
    "# IF3270 Pembelajaran Mesin | Praktikum 4\n"
    "## Recurrent Neural Network\n\n"
    "---"
))

cells.append(md(
    "**Nomor Kelompok:** K1-80\n\n"
    "**Anggota Kelompok:**\n"
    "- 13523002 - Refki Alfarizi\n"
    "- 13523004 - Razi Rachman Widyadhana"
))

# ---------------------------------------------------------------------------
# Daftar Isi
# ---------------------------------------------------------------------------
cells.append(md(
    "## Daftar Isi\n\n"
    "0. [**Inisialisasi**](#0)\n"
    "1. [**Muat Dataset**](#1)\n"
    "2. [**Analisis Data Eksploratif**](#2)\n"
    "3. [**Pembersihan Data**](#3)\n"
    "4. [**Prapemrosesan Data**](#4)\n"
    "5. [**Pemodelan dan Validasi**](#5)\n"
    "6. [**Analisis Kesalahan**](#6)"
))

# ---------------------------------------------------------------------------
# 0. Inisialisasi
# ---------------------------------------------------------------------------
cells.append(section("Inisialisasi", "0"))

cells.append(md("## Informasi Lingkungan"))
cells.append(code(
    "import sys, platform\n"
    "from pathlib import Path\n"
    "print('Python  :', sys.version)\n"
    "print('OS      :', platform.platform())\n"
    "print('CWD     :', Path.cwd())"
))

cells.append(md("## Instalasi Dependensi"))
cells.append(code(
    "# %pip install -q torch torchvision torchaudio\n"
    "# %pip install -q numpy pandas matplotlib seaborn scikit-learn tqdm"
))

cells.append(md("## Impor Pustaka"))
cells.append(code(
    "import math\n"
    "import os\n"
    "import platform\n"
    "import random\n"
    "import sys\n"
    "import warnings\n"
    "import zipfile\n"
    "from pathlib import Path\n\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from sklearn.metrics import mean_squared_error\n"
    "from sklearn.model_selection import TimeSeriesSplit\n"
    "from sklearn.preprocessing import MinMaxScaler\n"
    "from statsmodels.graphics.tsaplots import plot_acf\n"
    "from statsmodels.tsa.seasonal import seasonal_decompose\n"
    "from tqdm import tqdm\n\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from torch.utils.data import Dataset, DataLoader\n\n"
    "warnings.filterwarnings('ignore')\n"
    "print('PyTorch :', torch.__version__)\n"
    "print('CUDA    :', torch.version.cuda)\n"
    "print('Device  :', 'cuda' if torch.cuda.is_available() else 'cpu')"
))

cells.append(md("## Settings"))
cells.append(md(
    "Kelas `Settings` memusatkan seluruh konfigurasi eksperimen, mencakup jalur data "
    "dan *hyperparameter* pelatihan. Jalur data diselesaikan secara otomatis "
    "bergantung pada lingkungan eksekusi, yaitu Kaggle atau lokal. "
    "Lima parameter mengontrol perilaku modular, yaitu `INPUT_COLS` untuk fitur input "
    "(ubah ke beberapa kolom untuk mode *multivariate*), `VAL_STRATEGY` untuk strategi validasi, "
    "`INFER_MODE` yang menentukan cara prediksi data uji, `REGIME_FILTER` yang "
    "membatasi pelatihan hanya pada jendela *high-regime* (h1 > `REGIME_THRESHOLD`) sehingga "
    "*scaler* mendapatkan resolusi penuh pada rentang nilai yang relevan dengan data uji, "
    "serta `DIFF_MODE` yang mengubah target pelatihan dari nilai absolut menjadi selisih "
    "(`delta_h1 = h1[t+1] - h1[t]`) sehingga inferensi mengakumulasi prediksi perubahan "
    "ke dalam level dan menghindari konvergensi ke rata-rata. "
    "Tidak ada kode lain yang perlu diubah."
))
cells.append(code(
    "class Settings:\n"
    "    SEED   = 42\n\n"
    "    # Kolom data\n"
    "    TIME_COL   = 'time'\n"
    "    TARGET_COL = 'h1'\n"
    "    INPUT_COLS = ['h1']   # ganti ke ['WIND', 'RAIN', ...] untuk multivariate\n\n"
    "    # Arsitektur model\n"
    "    INPUT_SIZE  = len(INPUT_COLS)  # otomatis mengikuti INPUT_COLS\n"
    "    HIDDEN_SIZE = 64\n"
    "    NUM_LAYERS  = 2\n"
    "    DROPOUT     = 0.2\n\n"
    "    # Strategi inferensi\n"
    "    INFER_MODE  = 'autoregressive'  # 'autoregressive' | 'multistep'\n"
    "    HORIZON     = 1500               # langkah prediksi langsung (hanya saat INFER_MODE='multistep')\n"
    "    OUTPUT_SIZE = HORIZON if INFER_MODE == 'multistep' else 1\n\n"
    "    # Preprocessing\n"
    "    WINDOW_SIZE      = 7\n"
    "    REGIME_FILTER    = True    # True: latih hanya pada window high-regime\n"
    "    REGIME_THRESHOLD = 50000   # ambang batas h1 (asli) untuk high-regime\n"
    "    DIFF_MODE        = True    # True: target = delta(h1); inferensi akumulasi level\n\n"
    "    # Pelatihan\n"
    "    EPOCHS              = 100\n"
    "    BATCH_SIZE          = 32\n"
    "    LR                  = 1e-3\n"
    "    WEIGHT_DECAY        = 1e-4\n"
    "    EARLY_STOP_PATIENCE = 15\n\n"
    "    # Pembersihan data\n"
    "    CLEANING           = True  # kontrol deteksi anomali (flat-line, scale jump)\n"
    "    NAN_SHORT_THRESHOLD = 7    # run NaN <= ini diinterpolasi; lebih panjang di-forward fill + peringatan\n\n"
    "    # Strategi validasi\n"
    "    VAL_STRATEGY = 'holdout'  # 'holdout' | 'tscv'\n"
    "    VAL_SPLIT    = 0.2        # digunakan saat VAL_STRATEGY == 'holdout'\n"
    "    N_SPLITS     = 5          # jumlah fold (digunakan saat VAL_STRATEGY == 'tscv')\n\n"
    "    # Jalur data\n"
    "    _ON_KAGGLE   = Path('/kaggle/input').exists()\n"
    "    _LOCAL_DATA  = Path('data/univariate')\n"
    "    _KAGGLE_DATA = Path('/kaggle/input/praktikum-3-if-3270-ml')\n"
    "    DATA_DIR  = _KAGGLE_DATA if _ON_KAGGLE else _LOCAL_DATA\n"
    "    TRAIN_CSV = DATA_DIR / 'train.csv'\n"
    "    TEST_CSV  = DATA_DIR / 'test.csv'\n\n"
    "    # Jalur output\n"
    "    _BASE_OUT       = Path('/kaggle/working') if _ON_KAGGLE else Path('output')\n"
    "    OUTPUT_DIR      = _BASE_OUT / 'prak4'\n"
    "    CHECKPOINT_RNN  = OUTPUT_DIR / 'rnn_best.pth'\n"
    "    CHECKPOINT_LSTM = OUTPUT_DIR / 'lstm_best.pth'\n"
    "    SUBMISSION_RNN  = OUTPUT_DIR / 'submission_rnn.csv'\n"
    "    SUBMISSION_LSTM = OUTPUT_DIR / 'submission_lstm.csv'\n\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
    "print('Device :', DEVICE)\n"
    "print('Output :', CFG.OUTPUT_DIR)"
))

cells.append(md("## Pengaturan Seed"))
cells.append(code(
    "def seed_everything(seed: int = 42):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "    torch.backends.cudnn.deterministic = True\n"
    "    torch.backends.cudnn.benchmark = False\n\n"
    "seed_everything(CFG.SEED)"
))

# ---------------------------------------------------------------------------
# 1. Muat Dataset
# ---------------------------------------------------------------------------
cells.append(section("Muat Dataset", "1"))

cells.append(md(
    "Dataset terdiri dari dua berkas CSV, yaitu `train.csv` untuk pelatihan dan `test.csv` "
    "sebagai data uji kompetisi Kaggle. Kolom `time` dikonversi menjadi tipe *datetime* "
    "dan dijadikan indeks agar pemrosesan deret waktu lebih mudah dilakukan."
))
cells.append(code(
    "train_df = pd.read_csv(CFG.TRAIN_CSV, parse_dates=[CFG.TIME_COL], index_col=CFG.TIME_COL)\n"
    "test_df  = pd.read_csv(CFG.TEST_CSV,  parse_dates=[CFG.TIME_COL], index_col=CFG.TIME_COL)"
))

cells.append(md("## Pratinjau Data Latih"))
cells.append(md(
    "Lima baris pertama data latih memberikan gambaran awal tentang struktur kolom "
    "dan rentang nilai `h1`."
))
cells.append(code("train_df"))

# ---------------------------------------------------------------------------
# 2. Analisis Data Eksploratif
# ---------------------------------------------------------------------------
cells.append(section("Analisis Data Eksploratif", "2"))

cells.append(md(
    "Tahap ini memeriksa struktur dan karakteristik dataset sebelum dilakukan transformasi "
    "apa pun. Fokus diberikan pada ringkasan statistik dasar, pola deret waktu, dan "
    "identifikasi anomali yang dapat memengaruhi pelatihan model."
))

cells.append(md("## Analisis 1: Statistik Deskriptif"))
cells.append(md(
    "Ringkasan statistik memberikan gambaran distribusi nilai `h1`, mencakup rentang, "
    "rata-rata, dan indikasi persebaran data."
))
cells.append(code(
    "desc = train_df[CFG.TARGET_COL].describe()\n"
    "print(desc.to_string())"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi ringkasan statistik dan temuan awal di sini setelah notebook dijalankan."
))

cells.append(md("## Analisis 2: Visualisasi Deret Waktu"))
cells.append(md(
    "Grafik deret waktu memperlihatkan pola keseluruhan nilai `h1` sepanjang rentang data "
    "latih, sehingga anomali seperti loncatan skala atau segmen nilai konstan lebih mudah "
    "teridentifikasi secara visual."
))
cells.append(code(
    "fig, ax = plt.subplots(figsize=(14, 4))\n"
    "ax.plot(train_df.index, train_df[CFG.TARGET_COL], linewidth=0.6, color='steelblue')\n"
    "ax.set_title('Deret Waktu h1 (Data Latih)')\n"
    "ax.set_xlabel('Waktu')\n"
    "ax.set_ylabel('h1')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'eda_timeseries.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("### Deteksi Flat-Line (Nilai Konstan Berurutan)"))
cells.append(md(
    "Segmen dengan nilai identik selama beberapa hari berturut-turut berpotensi merupakan "
    "data yang diisi ke depan (*forward-filled*) akibat nilai yang hilang, bukan nilai "
    "pengukuran sesungguhnya."
))
cells.append(code(
    "def detect_flat_runs(series, min_len=3):\n"
    "    runs = []\n"
    "    start = 0\n"
    "    for i in range(1, len(series)):\n"
    "        if series.iloc[i] != series.iloc[i - 1]:\n"
    "            if i - start >= min_len:\n"
    "                runs.append({\n"
    "                    'start': series.index[start],\n"
    "                    'end':   series.index[i - 1],\n"
    "                    'length': i - start,\n"
    "                    'value': series.iloc[start],\n"
    "                })\n"
    "            start = i\n"
    "    if len(series) - start >= min_len:\n"
    "        runs.append({\n"
    "            'start': series.index[start],\n"
    "            'end':   series.index[-1],\n"
    "            'length': len(series) - start,\n"
    "            'value': series.iloc[start],\n"
    "        })\n"
    "    return pd.DataFrame(runs)\n\n"
    "flat_df = detect_flat_runs(train_df[CFG.TARGET_COL], min_len=3)\n"
    "print(f'Segmen flat-line ditemukan: {len(flat_df)}')\n"
    "display(flat_df)"
))

cells.append(md("### Deteksi Loncatan Skala"))
cells.append(code(
    "pct_change = train_df[CFG.TARGET_COL].pct_change().abs()\n"
    "threshold  = 5.0   # lebih dari 500% perubahan\n"
    "jumps = pct_change[pct_change > threshold].dropna()\n"
    "print(f'Loncatan besar (>500%) ditemukan: {len(jumps)}')\n"
    "if not jumps.empty:\n"
    "    display(train_df.loc[jumps.index, [CFG.TARGET_COL]])"
))

cells.append(md("## Analisis 3: Autokorelasi"))
cells.append(md(
    "Plot autokorelasi memperlihatkan seberapa kuat nilai `h1` pada waktu `t` "
    "berkorelasi dengan nilai pada `t - lag`. Puncak signifikan di lag 7 dan 365 "
    "mengonfirmasi adanya siklus mingguan dan tahunan, serta menjadi dasar "
    "penentuan ukuran *window* yang representatif."
))
cells.append(code(
    "series = train_df[CFG.TARGET_COL].dropna()\n"
    "key_lags = [7, 14, 30, 90, 182, 365, 730]\n"
    "acf_vals = {lag: round(series.autocorr(lag=lag), 4) for lag in key_lags}\n"
    "print('Autokorelasi pada lag kunci:')\n"
    "for lag, val in acf_vals.items():\n"
    "    print(f'  lag {lag:4d} : {val:+.4f}')\n\n"
    "fig, ax = plt.subplots(figsize=(14, 4))\n"
    "plot_acf(series, lags=730, ax=ax, alpha=0.05, zero=False)\n"
    "ax.set_title('Autokorelasi h1 (lag 1-730 hari)')\n"
    "ax.set_xlabel('Lag (hari)')\n"
    "ax.set_ylabel('Korelasi')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'eda_acf.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("## Analisis 4: Dekomposisi Musiman"))
cells.append(md(
    "Dekomposisi aditif memisahkan deret waktu menjadi tiga komponen, yaitu tren jangka "
    "panjang, musiman dengan periode 365 hari, dan residu. Komponen musiman yang kuat "
    "menandakan bahwa informasi kalender perlu diberikan secara eksplisit kepada model "
    "sebagai fitur tambahan."
))
cells.append(code(
    "decomp = seasonal_decompose(\n"
    "    train_df[CFG.TARGET_COL].dropna(),\n"
    "    model='additive',\n"
    "    period=365,\n"
    "    extrapolate_trend='freq',\n"
    ")\n\n"
    "var_seasonal  = decomp.seasonal.var()\n"
    "var_residual  = decomp.resid.dropna().var()\n"
    "var_total     = train_df[CFG.TARGET_COL].dropna().var()\n"
    "seasonal_str  = 1 - var_residual / (var_seasonal + var_residual)\n"
    "print(f'Rentang komponen musiman : {decomp.seasonal.min():.2f} --> {decomp.seasonal.max():.2f}')\n"
    "print(f'Rentang komponen tren    : {decomp.trend.min():.2f} --> {decomp.trend.max():.2f}')\n"
    "print(f'Rentang komponen residu  : {decomp.resid.dropna().min():.2f} --> {decomp.resid.dropna().max():.2f}')\n"
    "print(f'Kekuatan musiman (F_s)   : {seasonal_str:.4f}  (>0.6 = kuat, >0.4 = sedang)')\n"
    "print(f'Rasio varian musiman     : {var_seasonal / var_total * 100:.1f}% dari total varian')\n\n"
    "fig = decomp.plot()\n"
    "fig.set_size_inches(14, 9)\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'eda_decomposition.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("## Analisis 5: Pola Mingguan dan Tahunan"))
cells.append(md(
    "Distribusi `h1` dikelompokkan per hari dalam seminggu dan per bulan untuk "
    "mengukur seberapa besar variasi kalender memengaruhi beban listrik. Variasi "
    "yang jelas antara hari kerja dan akhir pekan, atau antara musim panas dan "
    "musim dingin, menjadi argumen kuat untuk menyertakan fitur `sin/cos` hari-dalam-tahun "
    "dan hari-dalam-seminggu sebagai input model."
))
cells.append(code(
    "tmp = train_df[[CFG.TARGET_COL]].copy()\n"
    "tmp['hari']  = tmp.index.dayofweek\n"
    "tmp['bulan'] = tmp.index.month\n\n"
    "day_labels   = ['Sen', 'Sel', 'Rab', 'Kam', 'Jum', 'Sab', 'Min']\n"
    "month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun',\n"
    "                'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']\n\n"
    "dow_stats = tmp.groupby('hari')[CFG.TARGET_COL].agg(['mean', 'median', 'std'])\n"
    "dow_stats.index = day_labels\n"
    "dow_stats.columns = ['Rata-rata', 'Median', 'Std']\n"
    "print('Statistik per Hari dalam Seminggu:')\n"
    "print(dow_stats.round(1).to_string())\n\n"
    "mon_stats = tmp.groupby('bulan')[CFG.TARGET_COL].agg(['mean', 'median', 'std'])\n"
    "mon_stats.index = month_labels\n"
    "mon_stats.columns = ['Rata-rata', 'Median', 'Std']\n"
    "print('\\nStatistik per Bulan:')\n"
    "print(mon_stats.round(1).to_string())\n\n"
    "print(f'\\nRentang rata-rata harian : {dow_stats[\"Rata-rata\"].min():.1f} - {dow_stats[\"Rata-rata\"].max():.1f}')\n"
    "print(f'Rentang rata-rata bulanan: {mon_stats[\"Rata-rata\"].min():.1f} - {mon_stats[\"Rata-rata\"].max():.1f}')\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "sns.boxplot(data=tmp, x='hari',  y=CFG.TARGET_COL, ax=axes[0], showfliers=False)\n"
    "axes[0].set_xticklabels(day_labels)\n"
    "axes[0].set_title('h1 per Hari dalam Seminggu')\n"
    "axes[0].set_xlabel('Hari')\n"
    "axes[0].set_ylabel('h1')\n\n"
    "sns.boxplot(data=tmp, x='bulan', y=CFG.TARGET_COL, ax=axes[1], showfliers=False)\n"
    "axes[1].set_xticklabels(month_labels)\n"
    "axes[1].set_title('h1 per Bulan')\n"
    "axes[1].set_xlabel('Bulan')\n"
    "axes[1].set_ylabel('h1')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'eda_seasonal_patterns.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi temuan EDA di sini setelah notebook dijalankan: anomali flat-line, loncatan "
    "> skala, kekuatan autokorelasi pada lag 7 dan 365, komponen musiman dari dekomposisi, "
    "> serta pola mingguan dan tahunan yang mengindikasikan perlunya fitur kalender."
))

# ---------------------------------------------------------------------------
# 3. Pembersihan Data
# ---------------------------------------------------------------------------
cells.append(section("Pembersihan Data", "3"))

cells.append(md(
    "Pembersihan terdiri dari dua lapis.\n\n"
    "- NaN di awal seri dipotong karena tidak ada nilai sebelumnya untuk diinterpolasi\n"
    "- Gap pendek (<= `NAN_SHORT_THRESHOLD`) diisi dengan interpolasi linear\n"
    "- Gap panjang diisi secara *forward fill*\n"
    "- Deteksi anomali flat-line dan loncatan skala dijalankan hanya jika `CFG.CLEANING = True`"
))

cells.append(md("## Penanganan NaN"))
cells.append(md(
    "Fungsi `handle_nan` menerapkan strategi bertingkat: baris di awal seri yang seluruhnya "
    "NaN dipotong karena tidak ada nilai sebelumnya untuk diinterpolasi, gap pendek diisi "
    "dengan interpolasi linear, dan gap panjang diisi secara *forward fill* sebagai fallback."
))
cells.append(code(
    "train_clean = train_df.copy()\n\n"
    "def handle_nan(series):\n"
    "    s = series.copy()\n"
    "    # 1. Potong leading NaN\n"
    "    first_valid = s.first_valid_index()\n"
    "    if first_valid is None:\n"
    "        print('[NaN] Seluruh seri adalah NaN.')\n"
    "        return s\n"
    "    if first_valid != s.index[0]:\n"
    "        n_cut = s.index.get_loc(first_valid)\n"
    "        print(f'[NaN] {n_cut} baris awal dipotong (leading NaN).')\n"
    "        s = s.loc[first_valid:]\n"
    "    if not s.isna().any():\n"
    "        print('[NaN] Tidak ada NaN.')\n"
    "        return s\n"
    "    # 2. Klasifikasi run NaN\n"
    "    null_mask  = s.isna()\n"
    "    run_id     = (null_mask != null_mask.shift()).cumsum()\n"
    "    run_len    = null_mask.groupby(run_id).transform('sum')\n"
    "    short_mask = null_mask & (run_len <= CFG.NAN_SHORT_THRESHOLD)\n"
    "    long_mask  = null_mask & (run_len >  CFG.NAN_SHORT_THRESHOLD)\n"
    "    print(f'[NaN] Run pendek (<={CFG.NAN_SHORT_THRESHOLD} hari): {short_mask.sum()} nilai -> interpolasi linear')\n"
    "    print(f'[NaN] Run panjang (>{CFG.NAN_SHORT_THRESHOLD} hari): {long_mask.sum()} nilai -> forward fill')\n"
    "    # 3. Interpolasi run pendek, forward fill run panjang\n"
    "    s = s.interpolate(method='linear', limit=CFG.NAN_SHORT_THRESHOLD)\n"
    "    s = s.ffill().bfill()\n"
    "    print(f'[NaN] Sisa NaN setelah semua strategi: {s.isna().sum()}')\n"
    "    return s\n\n"
    "train_clean[CFG.TARGET_COL] = handle_nan(train_clean[CFG.TARGET_COL])\n"
    "print(f'Shape setelah penanganan NaN: {train_clean.shape}')"
))

cells.append(md("## Pembersihan Anomali"))
cells.append(md(
    "Segmen flat-line dari EDA diganti dengan NaN lalu diinterpolasi secara linear. "
    "Loncatan skala hanya dilaporkan tanpa modifikasi karena merupakan karakteristik "
    "deret waktu yang harus dipelajari model."
))
cells.append(code(
    "if CFG.CLEANING:\n"
    "    # Ganti segmen flat-line dengan NaN lalu interpolasi\n"
    "    for _, row in flat_df.iterrows():\n"
    "        mask = (train_clean.index >= row['start']) & (train_clean.index <= row['end'])\n"
    "        train_clean.loc[mask, CFG.TARGET_COL] = np.nan\n"
    "    n_nan = train_clean[CFG.TARGET_COL].isna().sum()\n"
    "    print(f'NaN setelah penandaan flat-line : {n_nan}')\n"
    "    train_clean[CFG.TARGET_COL] = train_clean[CFG.TARGET_COL].interpolate(method='linear')\n"
    "    print(f'NaN setelah interpolasi         : {train_clean[CFG.TARGET_COL].isna().sum()}')\n"
    "    # Laporkan loncatan skala (tidak diubah; model perlu mempelajarinya)\n"
    "    pct_change = train_clean[CFG.TARGET_COL].pct_change().abs()\n"
    "    jumps = pct_change[pct_change > 5.0].dropna()\n"
    "    print(f'Loncatan skala besar (>500%) : {len(jumps)}')\n"
    "    if not jumps.empty:\n"
    "        display(train_clean.loc[jumps.index, [CFG.TARGET_COL]])\n"
    "else:\n"
    "    print('[Pembersihan] CLEANING=False, deteksi anomali dilewati.')"
))

cells.append(code(
    "fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)\n"
    "axes[0].plot(train_df.index, train_df[CFG.TARGET_COL],   linewidth=0.6, color='steelblue', label='Asli')\n"
    "axes[1].plot(train_clean.index, train_clean[CFG.TARGET_COL], linewidth=0.6, color='tomato',    label='Setelah Pembersihan')\n"
    "for ax in axes:\n"
    "    ax.set_ylabel('h1')\n"
    "    ax.legend()\n"
    "axes[0].set_title('Data Asli')\n"
    "axes[1].set_title('Data Setelah Pembersihan')\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'cleaning_comparison.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi evaluasi hasil pembersihan di sini setelah notebook dijalankan."
))

# ---------------------------------------------------------------------------
# 4. Prapemrosesan Data
# ---------------------------------------------------------------------------
cells.append(section("Prapemrosesan Data", "4"))

cells.append(md(
    "Prapemrosesan mencakup tiga langkah utama, yaitu *scaling* untuk menormalkan rentang nilai, "
    "*sliding window* untuk mengubah deret waktu menjadi pasangan input-target, "
    "dan pembagian data latih serta validasi tanpa pengacakan agar urutan waktu tetap terjaga. "
    "Strategi pembagian dikontrol oleh `CFG.VAL_STRATEGY`, di mana `'holdout'` menggunakan "
    "satu pembagian temporal 80/20 sedangkan `'tscv'` menerapkan *time series cross-validation* "
    "dengan `CFG.N_SPLITS` fold."
))

cells.append(md("## Scaling"))
cells.append(md(
    "`MinMaxScaler` dilatih hanya pada data latih untuk menghindari *data leakage*. "
    "Ketika `CFG.REGIME_FILTER=True`, *scaler* dilatih hanya pada baris *high-regime* "
    "(h1 > `REGIME_THRESHOLD`) sehingga rentang [0, 1] mencakup sepenuhnya variasi "
    "pada regime yang relevan dengan data uji. "
    "Skala yang sama kemudian diterapkan pada seluruh data latih."
))
cells.append(code(
    "scaler = MinMaxScaler(feature_range=(0, 1))\n"
    "train_values = train_clean[CFG.TARGET_COL].values.reshape(-1, 1)\n\n"
    "if CFG.REGIME_FILTER:\n"
    "    high_mask = train_clean[CFG.TARGET_COL] > CFG.REGIME_THRESHOLD\n"
    "    scaler.fit(train_values[high_mask.values])\n"
    "    print(f'[Regime] Scaler dilatih pada {high_mask.sum()} baris '\n"
    "          f'high-regime (h1 > {CFG.REGIME_THRESHOLD})')\n"
    "else:\n"
    "    scaler.fit(train_values)\n\n"
    "scaled_values = scaler.transform(train_values).flatten()\n\n"
    "train_scaled_df = pd.DataFrame(\n"
    "    {CFG.TARGET_COL: scaled_values},\n"
    "    index=train_clean.index\n"
    ")\n"
    "print('Rentang sebelum scaling :', train_values.min(), '-->', train_values.max())\n"
    "print('Rentang setelah scaling  :', scaled_values.min(), '-->', scaled_values.max())"
))

cells.append(md("## Sliding Window"))
cells.append(md(
    "Fungsi `sliding_window` mengubah deret waktu satu dimensi menjadi array tiga dimensi "
    "`(N, window_size, n_fitur)` untuk input model, serta array satu dimensi `(N,)` "
    "untuk target. Setiap sampel ke-i menggunakan nilai pada posisi i hingga i+L-1 "
    "sebagai input, dan nilai pada posisi i+L sebagai target."
))
cells.append(code(
    "def sliding_window(df, input_columns, target_column, time_steps):\n"
    "    if isinstance(input_columns, str):\n"
    "        input_columns = [input_columns]\n"
    "    X, y = [], []\n"
    "    n_rows = len(df) - time_steps - (CFG.OUTPUT_SIZE - 1)\n"
    "    for i in range(n_rows):\n"
    "        X.append(df[input_columns].iloc[i:i + time_steps].values)\n"
    "        if target_column is not None:\n"
    "            if CFG.OUTPUT_SIZE == 1:\n"
    "                y.append(df[target_column].iloc[i + time_steps])\n"
    "            else:\n"
    "                y.append(df[target_column].iloc[i + time_steps:i + time_steps + CFG.OUTPUT_SIZE].values)\n"
    "    return np.array(X), np.array(y)\n\n"
    "X_all, y_all = sliding_window(\n"
    "    train_scaled_df,\n"
    "    input_columns=CFG.INPUT_COLS,\n"
    "    target_column=CFG.TARGET_COL,\n"
    "    time_steps=CFG.WINDOW_SIZE,\n"
    ")\n\n"
    "orig_values = train_clean[CFG.TARGET_COL].values\n"
    "if CFG.REGIME_FILTER:\n"
    "    n_windows = len(orig_values) - CFG.WINDOW_SIZE\n"
    "    keep = np.array([\n"
    "        np.all(orig_values[i:i + CFG.WINDOW_SIZE] > CFG.REGIME_THRESHOLD)\n"
    "        and orig_values[i + CFG.WINDOW_SIZE] > CFG.REGIME_THRESHOLD\n"
    "        for i in range(n_windows)\n"
    "    ])\n"
    "    orig_positions = np.where(keep)[0]\n"
    "    X_all = X_all[keep]\n"
    "    y_all = y_all[keep]\n"
    "    print(f'[Regime] Windows high-regime: {keep.sum()} / {len(keep)}')\n"
    "else:\n"
    "    orig_positions = np.arange(len(X_all))\n\n"
    "if CFG.DIFF_MODE:\n"
    "    y_all = y_all - X_all[:, -1, 0]  # target -> delta dari nilai input terakhir\n"
    "    print('[Diff] Target diubah ke selisih (delta scaled h1)')\n\n"
    "print('X shape :', X_all.shape)\n"
    "print('y shape :', y_all.shape)"
))

cells.append(md("## Strategi Validasi dan Pembagian Data"))
cells.append(md(
    "Fungsi `get_splits` mengembalikan daftar tuple `(X_tr, y_tr, X_vl, y_vl, val_idx)` "
    "sesuai strategi yang dipilih. Untuk `'holdout'` daftar berisi satu elemen; "
    "untuk `'tscv'` berisi `N_SPLITS` elemen dengan ukuran latih yang semakin bertambah "
    "di setiap fold. `val_idx` menyimpan indeks asli ke `X_all` sehingga tanggal validasi "
    "dapat dipetakan kembali."
))
cells.append(code(
    "def get_splits(X, y):\n"
    "    if CFG.VAL_STRATEGY == 'holdout':\n"
    "        n = int(len(X) * (1 - CFG.VAL_SPLIT))\n"
    "        return [(X[:n], y[:n], X[n:], y[n:], np.arange(n, len(X)))]\n"
    "    tscv = TimeSeriesSplit(n_splits=CFG.N_SPLITS)\n"
    "    return [\n"
    "        (X[ti], y[ti], X[vi], y[vi], vi)\n"
    "        for ti, vi in tscv.split(X)\n"
    "    ]\n\n"
    "splits = get_splits(X_all, y_all)\n"
    "print(f'Strategi  : {CFG.VAL_STRATEGY}')\n"
    "print(f'Fold      : {len(splits)}')\n"
    "for i, (Xtr, ytr, Xvl, yvl, vi) in enumerate(splits):\n"
    "    d0 = train_scaled_df.index[orig_positions[vi[0]]  + CFG.WINDOW_SIZE].date()\n"
    "    d1 = train_scaled_df.index[orig_positions[vi[-1]] + CFG.WINDOW_SIZE].date()\n"
    "    print(f'  Fold {i+1}: train={len(Xtr):5d}  val={len(Xvl):5d}  '\n"
    "          f'val_range={d0} --> {d1}')"
))

cells.append(md("## Dataset PyTorch"))
cells.append(md(
    "Kelas `TimeSeriesDataset` membungkus array NumPy menjadi format yang dapat dikonsumsi "
    "oleh `DataLoader` PyTorch. DataLoader dibuat di dalam `run_cv` per fold agar ukuran "
    "latih dan validasi menyesuaikan strategi yang dipilih."
))
cells.append(code(
    "class TimeSeriesDataset(Dataset):\n"
    "    def __init__(self, X: np.ndarray, y: np.ndarray):\n"
    "        self.X = torch.FloatTensor(X)\n"
    "        self.y = torch.FloatTensor(y) if y.ndim > 1 else torch.FloatTensor(y).unsqueeze(-1)\n"
    "    def __len__(self):\n"
    "        return len(self.X)\n"
    "    def __getitem__(self, idx):\n"
    "        return self.X[idx], self.y[idx]"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi temuan prapemrosesan di sini: pengaruh scaling terhadap konvergensi, "
    "> pemilihan ukuran window, dan alasan pembagian temporal."
))

# ---------------------------------------------------------------------------
# 5. Pemodelan dan Validasi
# ---------------------------------------------------------------------------
cells.append(section("Pemodelan dan Validasi", "5"))

cells.append(md(
    "Dua arsitektur diimplementasikan dari nol menggunakan PyTorch: *Recurrent Neural Network* "
    "(RNN) vanilla dan *Long Short-Term Memory* (LSTM). Keduanya dilatih pada data yang sama "
    "dengan konfigurasi identik agar perbandingan adil, kemudian dievaluasi menggunakan "
    "RMSE pada skala asli nilai `h1`."
))

cells.append(md("## Utilitas Pelatihan"))
cells.append(md(
    "Fungsi-fungsi berikut digunakan bersama oleh model RNN dan LSTM:\n\n"
    "- `train_one_epoch`: pelatihan satu *epoch*\n"
    "- `evaluate`: evaluasi dengan kalkulasi RMSE pada skala asli\n"
    "- `run_training`: loop pelatihan lengkap dengan *early stopping* dan penyimpanan *checkpoint* terbaik\n"
    "- `predict_autoregressive`: prediksi berurutan (*autoregressive*) pada data uji\n"
    "- `predict_multistep`: prediksi langsung semua langkah dalam satu *forward pass*\n"
    "- `predict_test`: dispatcher berdasarkan `CFG.INFER_MODE`\n"
    "- `run_cv`: orkestrasi seluruh fold sesuai strategi validasi"
))
cells.append(code(
    "def train_one_epoch(model, loader, optimizer, criterion):\n"
    "    model.train()\n"
    "    total_loss = 0.0\n"
    "    for X_batch, y_batch in loader:\n"
    "        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)\n"
    "        optimizer.zero_grad()\n"
    "        preds = model(X_batch)\n"
    "        loss  = criterion(preds, y_batch)\n"
    "        loss.backward()\n"
    "        optimizer.step()\n"
    "        total_loss += loss.item() * len(y_batch)\n"
    "    return total_loss / len(loader.dataset)\n\n"
    "@torch.no_grad()\n"
    "def evaluate(model, loader, criterion):\n"
    "    model.eval()\n"
    "    total_loss = 0.0\n"
    "    all_preds, all_labels = [], []\n"
    "    for X_batch, y_batch in loader:\n"
    "        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)\n"
    "        preds = model(X_batch)\n"
    "        loss  = criterion(preds, y_batch)\n"
    "        total_loss += loss.item() * len(y_batch)\n"
    "        all_preds  += preds.cpu().numpy().flatten().tolist()\n"
    "        all_labels += y_batch.cpu().numpy().flatten().tolist()\n"
    "    avg_loss = total_loss / len(loader.dataset)\n"
    "    if CFG.DIFF_MODE:\n"
    "        scale       = float(scaler.data_max_[0] - scaler.data_min_[0])\n"
    "        preds_orig  = np.array(all_preds)  * scale\n"
    "        labels_orig = np.array(all_labels) * scale\n"
    "    else:\n"
    "        preds_orig  = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()\n"
    "        labels_orig = scaler.inverse_transform(np.array(all_labels).reshape(-1, 1)).flatten()\n"
    "    rmse = math.sqrt(mean_squared_error(labels_orig, preds_orig))\n"
    "    return avg_loss, rmse, preds_orig, labels_orig\n\n"
    "def run_training(model, train_loader, val_loader, checkpoint_path):\n"
    "    model.to(DEVICE)\n"
    "    optimizer  = torch.optim.Adam(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)\n"
    "    criterion  = nn.MSELoss()\n"
    "    best_rmse  = float('inf')\n"
    "    no_improve = 0\n"
    "    history    = []\n"
    "    for epoch in range(1, CFG.EPOCHS + 1):\n"
    "        tr_loss              = train_one_epoch(model, train_loader, optimizer, criterion)\n"
    "        vl_loss, vl_rmse, _, _ = evaluate(model, val_loader, criterion)\n"
    "        history.append({'epoch': epoch, 'tr_loss': tr_loss,\n"
    "                         'vl_loss': vl_loss, 'vl_rmse': vl_rmse})\n"
    "        print(f'Epoch {epoch:03d}/{CFG.EPOCHS}  '\n"
    "              f'train_loss={tr_loss:.6f}  '\n"
    "              f'val_loss={vl_loss:.6f}  '\n"
    "              f'val_rmse={vl_rmse:.2f}')\n"
    "        if vl_rmse < best_rmse:\n"
    "            best_rmse  = vl_rmse\n"
    "            no_improve = 0\n"
    "            torch.save(model.state_dict(), checkpoint_path)\n"
    "            print(f'  [OK] Checkpoint disimpan (val_rmse={best_rmse:.2f})')\n"
    "        else:\n"
    "            no_improve += 1\n"
    "            if no_improve >= CFG.EARLY_STOP_PATIENCE:\n"
    "                print(f'  [Early Stop] Tidak ada perbaikan selama {CFG.EARLY_STOP_PATIENCE} epoch.')\n"
    "                break\n"
    "    print(f'Val RMSE terbaik: {best_rmse:.2f}')\n"
    "    return history\n\n"
    "@torch.no_grad()\n"
    "def predict_autoregressive(model, seed_seq, n_steps):\n"
    "    model.eval()\n"
    "    history = list(seed_seq.copy())\n"
    "    preds   = []\n"
    "    for _ in range(n_steps):\n"
    "        window = np.array(history[-CFG.WINDOW_SIZE:]).reshape(1, CFG.WINDOW_SIZE, CFG.INPUT_SIZE)\n"
    "        x      = torch.FloatTensor(window).to(DEVICE)\n"
    "        out    = model(x).cpu().numpy().flatten()[0]\n"
    "        pred   = history[-1] + out if CFG.DIFF_MODE else out\n"
    "        preds.append(pred)\n"
    "        history.append(pred)\n"
    "    preds_orig = scaler.inverse_transform(\n"
    "        np.array(preds).reshape(-1, 1)\n"
    "    ).flatten()\n"
    "    return preds_orig\n\n"
    "@torch.no_grad()\n"
    "def predict_multistep(model, seed_seq):\n"
    "    model.eval()\n"
    "    window = np.array(seed_seq[-CFG.WINDOW_SIZE:]).reshape(1, CFG.WINDOW_SIZE, CFG.INPUT_SIZE)\n"
    "    x      = torch.FloatTensor(window).to(DEVICE)\n"
    "    preds  = model(x).cpu().numpy().flatten()  # shape (HORIZON,)\n"
    "    return scaler.inverse_transform(preds.reshape(-1, 1)).flatten()\n\n"
    "def predict_test(model, seed_seq, n_steps):\n"
    "    if CFG.INFER_MODE == 'multistep':\n"
    "        return predict_multistep(model, seed_seq)[:n_steps]\n"
    "    return predict_autoregressive(model, seed_seq, n_steps)\n\n"
    "def run_cv(model_class, checkpoint_path, label, positions):\n"
    "    fold_rmse  = []\n"
    "    last_hist  = []\n"
    "    last_preds = last_labels = last_dates = None\n"
    "    for fold, (X_tr, y_tr, X_vl, y_vl, val_idx) in enumerate(splits):\n"
    "        n_folds = len(splits)\n"
    "        print(f'\\n[{label}] Fold {fold + 1}/{n_folds}  '\n"
    "              f'(train={len(X_tr)}, val={len(X_vl)})')\n"
    "        tr_loader = DataLoader(\n"
    "            TimeSeriesDataset(X_tr, y_tr), batch_size=CFG.BATCH_SIZE, shuffle=False)\n"
    "        vl_loader = DataLoader(\n"
    "            TimeSeriesDataset(X_vl, y_vl), batch_size=CFG.BATCH_SIZE, shuffle=False)\n"
    "        seed_everything(CFG.SEED)\n"
    "        model   = model_class()\n"
    "        history = run_training(model, tr_loader, vl_loader, checkpoint_path)\n"
    "        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))\n"
    "        model.to(DEVICE)\n"
    "        _, rmse, _, _ = evaluate(model, vl_loader, nn.MSELoss())\n"
    "        fold_rmse.append(rmse)\n"
    "        last_hist = history\n"
    "        # End-to-end prediction from val start (consistent for both INFER_MODE)\n"
    "        val_start   = positions[val_idx[0]]\n"
    "        val_seed    = scaled_values[max(0, val_start - CFG.WINDOW_SIZE):val_start]\n"
    "        last_preds  = predict_test(model, val_seed, len(X_vl))\n"
    "        if CFG.DIFF_MODE:\n"
    "            y_flat      = y_vl[:, 0] if y_vl.ndim > 1 else y_vl\n"
    "            y_levels    = y_flat + X_vl[:, -1, 0]\n"
    "            last_labels = scaler.inverse_transform(y_levels.reshape(-1, 1)).flatten()\n"
    "        else:\n"
    "            last_labels = scaler.inverse_transform(\n"
    "                (y_vl[:, 0] if y_vl.ndim > 1 else y_vl).reshape(-1, 1)\n"
    "            ).flatten()\n"
    "        last_dates  = train_scaled_df.index[positions[val_idx] + CFG.WINDOW_SIZE]\n"
    "        print(f'[{label}] Fold {fold + 1} Val RMSE: {rmse:.4f}')\n"
    "    mean_rmse = float(np.mean(fold_rmse))\n"
    "    std_rmse  = float(np.std(fold_rmse))\n"
    "    print(f'\\n[{label}] Mean Val RMSE ({CFG.VAL_STRATEGY}): '\n"
    "          f'{mean_rmse:.4f} (+/- {std_rmse:.4f})')\n"
    "    final_model = model_class()\n"
    "    final_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))\n"
    "    final_model.to(DEVICE)\n"
    "    return fold_rmse, last_hist, last_preds, last_labels, last_dates, final_model"
))

cells.append(md("## 5.1 Recurrent Neural Network"))
cells.append(md(
    "RNN vanilla memproses setiap langkah waktu secara berurutan dan meneruskan *hidden state* "
    "ke langkah berikutnya. Arsitektur ini sederhana namun rentan terhadap masalah "
    "*vanishing gradient* pada sekuens panjang."
))

cells.append(md("### 5.1.1 Model RNN dari Nol"))
cells.append(code(
    "class RNNModel(nn.Module):\n"
    "    def __init__(\n"
    "        self,\n"
    "        input_size=CFG.INPUT_SIZE,\n"
    "        hidden_size=CFG.HIDDEN_SIZE,\n"
    "        num_layers=CFG.NUM_LAYERS,\n"
    "        dropout=CFG.DROPOUT,\n"
    "        output_size=CFG.OUTPUT_SIZE,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.rnn = nn.RNN(\n"
    "            input_size, hidden_size, num_layers,\n"
    "            batch_first=True,\n"
    "            dropout=dropout if num_layers > 1 else 0.0,\n"
    "        )\n"
    "        self.fc = nn.Linear(hidden_size, output_size)\n\n"
    "    def forward(self, x):\n"
    "        out, _ = self.rnn(x)\n"
    "        out = out[:, -1, :]  # ambil output langkah terakhir\n"
    "        return self.fc(out)\n\n"
    "rnn_model = RNNModel()\n"
    "print(rnn_model)\n"
    "print('Parameter RNN:', sum(p.numel() for p in rnn_model.parameters()))"
))

cells.append(md("### Pelatihan RNN"))
cells.append(code(
    "rnn_fold_rmse, rnn_history, rnn_val_preds, rnn_val_labels, rnn_val_dates, rnn_model = \\\n"
    "    run_cv(RNNModel, CFG.CHECKPOINT_RNN, 'RNN', orig_positions)\n"
    "rnn_val_rmse = float(np.mean(rnn_fold_rmse))\n"
    "print(f'RNN Val RMSE (mean): {rnn_val_rmse:.4f}')"
))

cells.append(md("### Kurva Pembelajaran RNN"))
cells.append(code(
    "hist_df = pd.DataFrame(rnn_history)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='Train Loss')\n"
    "axes[0].plot(hist_df['epoch'], hist_df['vl_loss'], label='Val Loss')\n"
    "axes[0].set_title('RNN - Loss')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].legend()\n"
    "axes[1].plot(hist_df['epoch'], hist_df['vl_rmse'], color='tomato', label='Val RMSE')\n"
    "axes[1].set_title('RNN - Val RMSE')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'rnn_learning_curve.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("## 5.2 Long Short-Term Memory"))
cells.append(md(
    "LSTM memperluas RNN dengan menambahkan mekanisme *gate* (input, forget, output) "
    "dan *cell state* sebagai memori jangka panjang. Mekanisme ini dirancang untuk "
    "mengatasi *vanishing gradient* dan memungkinkan model mengingat dependensi jangka jauh."
))

cells.append(md("### 5.2.1 Model LSTM dari Nol"))
cells.append(code(
    "class LSTMModel(nn.Module):\n"
    "    def __init__(\n"
    "        self,\n"
    "        input_size=CFG.INPUT_SIZE,\n"
    "        hidden_size=CFG.HIDDEN_SIZE,\n"
    "        num_layers=CFG.NUM_LAYERS,\n"
    "        dropout=CFG.DROPOUT,\n"
    "        output_size=CFG.OUTPUT_SIZE,\n"
    "    ):\n"
    "        super().__init__()\n"
    "        self.lstm = nn.LSTM(\n"
    "            input_size, hidden_size, num_layers,\n"
    "            batch_first=True,\n"
    "            dropout=dropout if num_layers > 1 else 0.0,\n"
    "        )\n"
    "        self.fc = nn.Linear(hidden_size, output_size)\n\n"
    "    def forward(self, x):\n"
    "        out, _ = self.lstm(x)\n"
    "        out = out[:, -1, :]  # ambil output langkah terakhir\n"
    "        return self.fc(out)\n\n"
    "lstm_model = LSTMModel()\n"
    "print(lstm_model)\n"
    "print('Parameter LSTM:', sum(p.numel() for p in lstm_model.parameters()))"
))

cells.append(md("### Pelatihan LSTM"))
cells.append(code(
    "lstm_fold_rmse, lstm_history, lstm_val_preds, lstm_val_labels, lstm_val_dates, lstm_model = \\\n"
    "    run_cv(LSTMModel, CFG.CHECKPOINT_LSTM, 'LSTM', orig_positions)\n"
    "lstm_val_rmse = float(np.mean(lstm_fold_rmse))\n"
    "print(f'LSTM Val RMSE (mean): {lstm_val_rmse:.4f}')"
))

cells.append(md("### Kurva Pembelajaran LSTM"))
cells.append(code(
    "hist_df = pd.DataFrame(lstm_history)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='Train Loss')\n"
    "axes[0].plot(hist_df['epoch'], hist_df['vl_loss'], label='Val Loss')\n"
    "axes[0].set_title('LSTM - Loss')\n"
    "axes[0].set_xlabel('Epoch')\n"
    "axes[0].legend()\n"
    "axes[1].plot(hist_df['epoch'], hist_df['vl_rmse'], color='tomato', label='Val RMSE')\n"
    "axes[1].set_title('LSTM - Val RMSE')\n"
    "axes[1].set_xlabel('Epoch')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'lstm_learning_curve.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("## Validasi: Perbandingan RNN dan LSTM"))
cells.append(md(
    "Kedua model dibandingkan berdasarkan RMSE pada set validasi. "
    "Tabel berikut merangkum hasil kuantitatif, diikuti visualisasi prediksi terhadap "
    "nilai aktual pada rentang waktu validasi."
))
cells.append(code(
    "n_folds = len(splits)\n"
    "print('=' * 50)\n"
    "print(f'  Strategi      : {CFG.VAL_STRATEGY} ({n_folds} fold)')\n"
    "print(f'  RNN  Val RMSE : {rnn_val_rmse:.4f}  (fold: {[round(r,2) for r in rnn_fold_rmse]})')\n"
    "print(f'  LSTM Val RMSE : {lstm_val_rmse:.4f}  (fold: {[round(r,2) for r in lstm_fold_rmse]})')\n"
    "print('=' * 50)\n\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)\n"
    "for ax, preds, labels, dates, label, color in zip(\n"
    "    axes,\n"
    "    [rnn_val_preds,  lstm_val_preds],\n"
    "    [rnn_val_labels, lstm_val_labels],\n"
    "    [rnn_val_dates,  lstm_val_dates],\n"
    "    ['RNN', 'LSTM'],\n"
    "    ['steelblue', 'darkorange'],\n"
    "):\n"
    "    ax.plot(dates, labels, label='Aktual',           color='gray',  linewidth=0.8)\n"
    "    ax.plot(dates, preds,  label=f'Prediksi {label}', color=color,  linewidth=0.8)\n"
    "    ax.set_title(f'{label} - Prediksi vs Aktual (Fold Terakhir)')\n"
    "    ax.set_ylabel('h1')\n"
    "    ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'validation_comparison.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md("## Submisi Kaggle"))
cells.append(md(
    "Prediksi data uji dihasilkan menggunakan strategi yang dikontrol `CFG.INFER_MODE`. "
    "Pada mode `'multistep'`, model memprediksi seluruh `HORIZON` langkah sekaligus dalam "
    "satu *forward pass* dari *seed* `WINDOW_SIZE` nilai terakhir data latih sehingga tidak "
    "ada akumulasi kesalahan antarlangkah. Pada mode `'autoregressive'`, setiap prediksi "
    "dimasukkan kembali sebagai input untuk langkah berikutnya. "
    "Dua berkas submisi dihasilkan, satu untuk setiap arsitektur."
))
cells.append(code(
    "# Seed: WINDOW_SIZE nilai terakhir dari data latih (skala ternormalisasi)\n"
    "seed_seq = scaled_values[-CFG.WINDOW_SIZE:]\n"
    "n_test   = len(test_df)\n\n"
    "rnn_test_preds  = predict_test(rnn_model,  seed_seq, n_test)\n"
    "lstm_test_preds = predict_test(lstm_model, seed_seq, n_test)\n\n"
    "print(f'Prediksi uji RNN  - min: {rnn_test_preds.min():.2f}, max: {rnn_test_preds.max():.2f}')\n"
    "print(f'Prediksi uji LSTM - min: {lstm_test_preds.min():.2f}, max: {lstm_test_preds.max():.2f}')"
))

cells.append(code(
    "submission_rnn = pd.DataFrame({\n"
    "    CFG.TIME_COL: test_df.index,\n"
    "    CFG.TARGET_COL: rnn_test_preds,\n"
    "})\n"
    "submission_rnn.to_csv(CFG.SUBMISSION_RNN, index=False)\n"
    "print('Submisi RNN  disimpan:', CFG.SUBMISSION_RNN)\n"
    "display(submission_rnn.head())\n\n"
    "submission_lstm = pd.DataFrame({\n"
    "    CFG.TIME_COL: test_df.index,\n"
    "    CFG.TARGET_COL: lstm_test_preds,\n"
    "})\n"
    "submission_lstm.to_csv(CFG.SUBMISSION_LSTM, index=False)\n"
    "print('Submisi LSTM disimpan:', CFG.SUBMISSION_LSTM)\n"
    "display(submission_lstm.head())"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi perbandingan kinerja RNN vs LSTM dan komentar atas hasil submisi Kaggle di sini."
))

# ---------------------------------------------------------------------------
# 6. Analisis Kesalahan
# ---------------------------------------------------------------------------
cells.append(section("Analisis Kesalahan", "6"))

cells.append(md(
    "Analisis kesalahan membandingkan prediksi model dengan nilai aktual pada set validasi "
    "secara kualitatif maupun kuantitatif. Visualisasi grafik garis memperlihatkan deviasi "
    "prediksi dari realisasi sepanjang waktu, sementara distribusi residual mengungkap "
    "pola kesalahan sistematis."
))

cells.append(md("## Analisis Kesalahan RNN"))
cells.append(md(
    "Residual dihitung sebagai selisih nilai aktual dikurangi prediksi, kemudian "
    "divisualisasikan terhadap waktu untuk mengidentifikasi periode dengan kesalahan tinggi."
))
cells.append(code(
    "rnn_residuals = rnn_val_labels - rnn_val_preds\n\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 8))\n\n"
    "axes[0].plot(rnn_val_dates, rnn_val_labels, label='Aktual',      color='gray',      linewidth=0.8)\n"
    "axes[0].plot(rnn_val_dates, rnn_val_preds,  label='Prediksi RNN', color='steelblue', linewidth=0.8)\n"
    "axes[0].set_title('RNN - Prediksi vs Aktual')\n"
    "axes[0].set_ylabel('h1')\n"
    "axes[0].legend()\n\n"
    "axes[1].bar(rnn_val_dates, rnn_residuals, color='steelblue', alpha=0.6, width=1)\n"
    "axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
    "axes[1].set_title('RNN - Residual (Aktual - Prediksi)')\n"
    "axes[1].set_ylabel('Residual')\n"
    "axes[1].set_xlabel('Waktu')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'error_rnn.png', dpi=120)\n"
    "plt.show()\n\n"
    "print(f'RNN Residual - Mean: {rnn_residuals.mean():.2f}, Std: {rnn_residuals.std():.2f}')"
))

cells.append(md("## Analisis Kesalahan LSTM"))
cells.append(md(
    "Analisis yang sama diterapkan pada model LSTM untuk memungkinkan perbandingan "
    "langsung pola kesalahan antara kedua arsitektur."
))
cells.append(code(
    "lstm_residuals = lstm_val_labels - lstm_val_preds\n\n"
    "fig, axes = plt.subplots(2, 1, figsize=(14, 8))\n\n"
    "axes[0].plot(lstm_val_dates, lstm_val_labels, label='Aktual',       color='gray',       linewidth=0.8)\n"
    "axes[0].plot(lstm_val_dates, lstm_val_preds,  label='Prediksi LSTM', color='darkorange', linewidth=0.8)\n"
    "axes[0].set_title('LSTM - Prediksi vs Aktual')\n"
    "axes[0].set_ylabel('h1')\n"
    "axes[0].legend()\n\n"
    "axes[1].bar(lstm_val_dates, lstm_residuals, color='darkorange', alpha=0.6, width=1)\n"
    "axes[1].axhline(0, color='black', linewidth=0.8, linestyle='--')\n"
    "axes[1].set_title('LSTM - Residual (Aktual - Prediksi)')\n"
    "axes[1].set_ylabel('Residual')\n"
    "axes[1].set_xlabel('Waktu')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'error_lstm.png', dpi=120)\n"
    "plt.show()\n\n"
    "print(f'LSTM Residual - Mean: {lstm_residuals.mean():.2f}, Std: {lstm_residuals.std():.2f}')"
))

cells.append(md("## Distribusi Residual"))
cells.append(md(
    "Distribusi residual menunjukkan apakah kesalahan model terpusat di sekitar nol "
    "atau memiliki bias sistematis ke arah tertentu."
))
cells.append(code(
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].hist(rnn_residuals,  bins=50, color='steelblue',  alpha=0.7, edgecolor='white')\n"
    "axes[0].axvline(0, color='black', linestyle='--')\n"
    "axes[0].set_title('Distribusi Residual RNN')\n"
    "axes[0].set_xlabel('Residual')\n"
    "axes[0].set_ylabel('Frekuensi')\n\n"
    "axes[1].hist(lstm_residuals, bins=50, color='darkorange', alpha=0.7, edgecolor='white')\n"
    "axes[1].axvline(0, color='black', linestyle='--')\n"
    "axes[1].set_title('Distribusi Residual LSTM')\n"
    "axes[1].set_xlabel('Residual')\n"
    "axes[1].set_ylabel('Frekuensi')\n\n"
    "plt.tight_layout()\n"
    "plt.savefig(CFG.OUTPUT_DIR / 'error_residual_dist.png', dpi=120)\n"
    "plt.show()"
))

cells.append(md(
    "> ### Insights\n"
    "> Isi analisis pola kesalahan di sini: apakah ada bias sistematis, periode waktu "
    "> tertentu yang sulit diprediksi, dan perbedaan karakteristik kesalahan RNN vs LSTM."
))

# ---------------------------------------------------------------------------
# Bundle Output
# ---------------------------------------------------------------------------
cells.append(section("Bundle Output", "bundle"))
cells.append(md(
    "Seluruh berkas hasil (model, kurva pembelajaran, visualisasi, dan submisi) "
    "dikemas menjadi satu arsip ZIP untuk kemudahan pengunduhan dari Kaggle."
))
cells.append(code(
    "bundle_path = CFG.OUTPUT_DIR / 'figures_bundle.zip'\n"
    "file_exts   = ['.png', '.pth', '.csv']\n\n"
    "with zipfile.ZipFile(bundle_path, 'w', zipfile.ZIP_DEFLATED) as zf:\n"
    "    for f in sorted(CFG.OUTPUT_DIR.iterdir()):\n"
    "        if f.suffix in file_exts and f != bundle_path:\n"
    "            zf.write(f, f.name)\n"
    "            print('[OK]', f.name)\n\n"
    "print('Bundle disimpan:', bundle_path)"
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells to {OUT}")
