#!/usr/bin/env python3
"""
Generate Prak-3/notebook.ipynb
IF3270 Pembelajaran Mesin | Praktikum 3 | Convolutional Neural Network

Run from Prak-3/ directory:
    python build_prak3.py
"""

import json
import uuid
from pathlib import Path

OUT = Path(__file__).parent / "notebook.ipynb"


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
    return md(f"---\n\n# {title} <a name=\"{anchor}\"></a>\n\n---")


# Sel-sel notebook
cells = []

# Banner
cells.append(md(
    "---\n\n"
    "# IF3270 Pembelajaran Mesin | Praktikum 3\n"
    "## Convolutional Neural Network\n\n"
    "---"
))

# Informasi Kelompok
cells.append(md(
    "**Nomor Kelompok:** xx\n\n"
    "**Anggota Kelompok:**\n"
    "- Nama (NIM)\n"
    "- Nama (NIM)"
))

# Daftar Isi
cells.append(md(
    "## Daftar Isi\n\n"
    "0. [**Inisialisasi**](#0)\n"
    "1. [**Analisis Data Eksploratif**](#1)\n"
    "2. [**Pembagian Data Latih dan Validasi**](#2)\n"
    "3. [**Pembersihan dan Prapemrosesan Data**](#3)\n"
    "4. [**Pemodelan dan Validasi**](#4)\n"
    "5. [**Analisis Kesalahan**](#5)\n"
    "6. [**Insights**](#6)"
))

# 0. Inisialisasi
cells.append(section("Inisialisasi", "0"))

cells.append(md("## Instalasi Dependensi"))
cells.append(code(
    "%pip install -q gdown kaggle torch torchvision torchaudio\n"
    "%pip install -q opencv-python-headless Pillow scikit-learn matplotlib seaborn tqdm"
))

cells.append(md("## Impor Pustaka"))
cells.append(code(
    "from collections import Counter\n"
    "import hashlib\n"
    "import os\n"
    "import platform\n"
    "import random\n"
    "import shutil\n"
    "import subprocess\n"
    "import sys\n"
    "import zipfile\n"
    "import warnings\n"
    "from pathlib import Path\n\n"
    "import gdown\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "from PIL import Image\n"
    "from tqdm import tqdm\n\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "from torch.utils.data import Dataset, DataLoader\n"
    "import torchvision.models as tv_models\n"
    "import torchvision.transforms as transforms\n"
    "from sklearn.model_selection import train_test_split\n"
    "from sklearn.metrics import f1_score, classification_report, confusion_matrix\n\n"
    "warnings.filterwarnings('ignore')\n"
    "print('PyTorch :', torch.__version__)\n"
    "print('CUDA    :', torch.version.cuda)\n"
    "print('Device  :', 'cuda' if torch.cuda.is_available() else 'cpu')"
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
    "seed_everything(42)"
))

cells.append(md("## Settings"))
cells.append(md(
    "Kelas `Settings` memusatkan seluruh konfigurasi eksperimen, mencakup jalur data, "
    "*hyperparameter* pelatihan, dan flag mode. Jalur data diselesaikan secara otomatis "
    "bergantung pada lingkungan eksekusi, yaitu Kaggle atau lokal. "
    "Ubah `MODE` menjadi `'inference'` untuk melewati pelatihan dan langsung memuat "
    "*checkpoint* tersimpan."
))
cells.append(code(
    "class Settings:\n"
    "    SEED = 42\n"
    "    MODE = 'train'   # 'train' | 'inference'\n\n"
    "    # Kelas target, urutan harus sesuai encoding integer di CSV\n"
    "    TARGET_COL  = 'Species_Label'\n"
    "    CLASS_NAMES = ['cat', 'dog']   # index 0 = kucing, 1 = anjing\n"
    "    N_CLASSES   = len(CLASS_NAMES)\n"
    "    LABEL2IDX   = {name: i for i, name in enumerate(CLASS_NAMES)}\n\n"
    "    # Pelatihan\n"
    "    EPOCHS      = 30\n"
    "    BATCH_SIZE  = 32\n"
    "    LR_ALEX          = 1e-4    # LR AlexNet dari nol\n"
    "    LR_HEAD          = 1e-3    # LR kepala pretrained (fase warm-up & fine-tune)\n"
    "    LR_BACKBONE      = 1e-4    # LR backbone pada fase fine-tune\n"
    "    WARMUP_PATIENCE      = 2   # toleransi epoch tanpa perbaikan val_loss di fase warm-up\n"
    "    WARMUP_MIN_DELTA     = 1e-3  # ambang perbaikan val_loss yang dianggap signifikan\n"
    "    EARLY_STOP_PATIENCE  = 5   # toleransi epoch tanpa perbaikan val_f1 sebelum berhenti\n"
    "    UNFREEZE_BLOCKS  = 2       # jumlah blok terakhir backbone yang di-unfreeze\n"
    "    COSINE_T_MAX     = 15      # T_max CosineAnnealingLR di fase 2 (tetap, tidak bergantung EPOCHS)\n"
    "    WEIGHT_DECAY = 1e-4\n"
    "    VAL_SPLIT   = 0.2\n"
    "    USE_AMP     = True\n"
    "    IMG_SIZE    = (224, 224)\n\n"
    "    # Normalisasi (statistik ImageNet)\n"
    "    IMAGENET_MEAN = (0.485, 0.456, 0.406)\n"
    "    IMAGENET_STD  = (0.229, 0.224, 0.225)\n\n"
    "    PRETRAINED_MODEL = 'resnet152'\n\n"
    "    # Visualisasi\n"
    "    PALETTE = 'flare'\n\n"
    "    # DataLoader\n"
    "    NUM_WORKERS = 1\n\n"
    "    # Jalur data, diselesaikan otomatis berdasarkan lingkungan eksekusi\n"
    "    _ON_KAGGLE  = Path('/kaggle/input').exists()\n"
    "    _LOCAL_DATA = Path('dataset')\n"
    "    _KAGGLE_DATA = Path('/kaggle/input/competitions/praktikum-2-if-3270-ml')\n\n"
    "    DATA_DIR   = _KAGGLE_DATA if _ON_KAGGLE else _LOCAL_DATA\n"
    "    IMAGES_DIR     = DATA_DIR / 'images' / 'images'\n"
    "    TRAIN_IMG_DIR  = IMAGES_DIR / 'train'\n"
    "    TEST_IMG_DIR   = IMAGES_DIR / 'test'\n"
    "    TRAIN_CSV  = DATA_DIR / 'train.csv'\n"
    "    TEST_CSV   = DATA_DIR / 'test.csv'\n\n"
    "    _BASE_OUT = Path('/kaggle/working') if _ON_KAGGLE else Path('output')\n"
    "    OUTPUT_DIR      = _BASE_OUT / 'prak3'\n"
    "    CHECKPOINT_ALEX = OUTPUT_DIR / 'alexnet_best.pth'\n"
    "    CHECKPOINT_FT   = OUTPUT_DIR / 'finetune_best.pth'\n"
    "    SUBMISSION_ALEX = OUTPUT_DIR / 'submission_alexnet.csv'\n"
    "    SUBMISSION_FT   = OUTPUT_DIR / 'submission_finetune.csv'\n\n\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "FIG_DIR = CFG.OUTPUT_DIR / 'figures'\n"
    "FIG_DIR.mkdir(parents=True, exist_ok=True)\n\n"
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n\n"
    "print(f'Python      : {sys.version.split()[0]}')\n"
    "print(f'OS          : {platform.system()} {platform.release()}')\n"
    "print(f'CWD         : {os.getcwd()}')\n"
    "print(f'Lingkungan  : {\"Kaggle\" if CFG._ON_KAGGLE else \"Lokal\"}')\n"
    "print(f'DATA_DIR    : {CFG.DATA_DIR}  exists={CFG.DATA_DIR.exists()}')\n"
    "print(f'TRAIN_IMG_DIR: {CFG.TRAIN_IMG_DIR}  exists={CFG.TRAIN_IMG_DIR.exists()}')\n"
    "print(f'TEST_IMG_DIR : {CFG.TEST_IMG_DIR}  exists={CFG.TEST_IMG_DIR.exists()}')\n"
    "print(f'OUTPUT_DIR  : {CFG.OUTPUT_DIR}')\n"
    "print(f'DEVICE      : {DEVICE}')"
))

cells.append(md("## Unduh Dataset"))
cells.append(md(
    "Apabila direktori *dataset* tidak ditemukan, sel ini mengunduhnya secara otomatis. "
    "*Download* dicoba terlebih dahulu melalui *gdown* (Google Drive); bila gagal, "
    "Kaggle API digunakan sebagai cadangan. Setelah diunduh, arsip zip diekstraksi "
    "dan file sementara dihapus."
))
cells.append(code(
    "# Konfigurasi gdown\n"
    "GDRIVE_FILE_ID = '1bY1il9qQiE1dHFRgriPiuumakfgUuuyC'  # perbarui jika URL berubah\n"
    "KAGGLE_COMPETITION = 'praktikum-2-if-3270-ml'\n\n"
    "def _extract(zip_path: Path, dest: Path) -> None:\n"
    "    print(f'Mengekstrak {zip_path.name} ...')\n"
    "    with zipfile.ZipFile(zip_path, 'r') as zf:\n"
    "        zf.extractall(dest)\n"
    "    zip_path.unlink()\n"
    "    print('[OK] Ekstraksi selesai')\n\n"
    "def _download_gdown(dest_dir: Path) -> bool:\n"
    "    try:\n"
    "        import gdown\n"
    "        url = f'https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}'\n"
    "        zip_path = dest_dir / 'dataset.zip'\n"
    "        dest_dir.mkdir(parents=True, exist_ok=True)\n"
    "        gdown.download(url, str(zip_path), quiet=False)\n"
    "        _extract(zip_path, dest_dir)\n"
    "        # tangani zip bersarang (images.zip)\n"
    "        inner = dest_dir / 'images.zip'\n"
    "        if inner.exists():\n"
    "            _extract(inner, dest_dir)\n"
    "        return True\n"
    "    except Exception as e:\n"
    "        print(f'[gdown gagal] {e}')\n"
    "        return False\n\n"
    "def _download_kaggle(dest_dir: Path) -> bool:\n"
    "    try:\n"
    "        import subprocess\n"
    "        # kredensial Kaggle harus berada di ~/.kaggle/kaggle.json atau variabel KAGGLE_*\n"
    "        dest_dir.mkdir(parents=True, exist_ok=True)\n"
    "        result = subprocess.run(\n"
    "            ['kaggle', 'competitions', 'download', '-c', KAGGLE_COMPETITION,\n"
    "             '-p', str(dest_dir)],\n"
    "            capture_output=True, text=True\n"
    "        )\n"
    "        print(result.stdout)\n"
    "        if result.returncode != 0:\n"
    "            raise RuntimeError(result.stderr)\n"
    "        for zip_path in dest_dir.glob('*.zip'):\n"
    "            _extract(zip_path, dest_dir)\n"
    "        return True\n"
    "    except Exception as e:\n"
    "        print(f'[Kaggle API gagal] {e}')\n"
    "        return False\n\n"
    "# Deteksi otomatis dan unduh\n"
    "if not CFG.TRAIN_IMG_DIR.exists():\n"
    "    print(f'Dataset tidak ditemukan di {CFG.DATA_DIR}. Mencoba mengunduh...')\n"
    "    ok = _download_gdown(CFG.DATA_DIR)\n"
    "    if not ok:\n"
    "        print('Mencoba ulang dengan Kaggle API...')\n"
    "        ok = _download_kaggle(CFG.DATA_DIR)\n"
    "    if not ok:\n"
    "        raise RuntimeError(\n"
    "            'Unduhan dataset gagal. Letakkan dataset secara manual di: ' + str(CFG.DATA_DIR)\n"
    "        )\n"
    "else:\n"
    "    print(f'[OK] Dataset ditemukan di {CFG.TRAIN_IMG_DIR}')"
))

cells.append(md("## Muat Dataset"))
cells.append(code(
    "train_df = pd.read_csv(CFG.TRAIN_CSV)\n"
    "test_df  = pd.read_csv(CFG.TEST_CSV)\n\n"
    "# Encode label ke integer jika kolom bertipe string\n"
    "if train_df[CFG.TARGET_COL].dtype == object:\n"
    "    train_df[CFG.TARGET_COL] = train_df[CFG.TARGET_COL].map(CFG.LABEL2IDX)\n"
    "    test_df[CFG.TARGET_COL]  = test_df[CFG.TARGET_COL].map(CFG.LABEL2IDX)\n"
    "    print(f'Label dienkode: {CFG.LABEL2IDX}')\n\n"
    "print(f'Sampel latih : {len(train_df)}')\n"
    "print(f'Sampel uji   : {len(test_df)}')\n"
    "train_df.head()"
))

# 1. Analisis Data Eksploratif
cells.append(section("Analisis Data Eksploratif", "1"))

cells.append(md(
    "*Exploratory Data Analysis* (EDA) adalah langkah awal sebelum menerapkan teknik "
    "*machine learning*, yang melibatkan pemeriksaan dan visualisasi *dataset* untuk "
    "mengungkap pola, tren, anomali, dan wawasan penting. "
    "Minimal **2 analisis** disertakan, masing-masing dijawab dengan visualisasi."
))

cells.append(md(
    "## Analisis 1: Pengecekan File Hilang\n\n"
    "> Apakah semua entri di CSV memiliki file gambar yang tersedia di disk?"
))
cells.append(md(
    "Ketidakcocokan antara entri CSV dan file gambar yang ada di disk menyebabkan error saat "
    "pelatihan. File yang hilang diidentifikasi terlebih dahulu, kemudian baris yang tidak "
    "memiliki gambar dihapus dari dataframe sebelum analisis berikutnya dilanjutkan."
))
cells.append(code(
    "def find_missing(df, images_dir):\n"
    "    return [fid for fid in df['Image'] if not (images_dir / fid).exists()]\n\n"
    "train_missing = find_missing(train_df, CFG.TRAIN_IMG_DIR)\n"
    "test_missing  = find_missing(test_df,  CFG.TEST_IMG_DIR)\n\n"
    "print(f'Train  total : {len(train_df)}')\n"
    "print(f'Train  hilang: {len(train_missing)}')\n"
    "print(f'Test   total : {len(test_df)}')\n"
    "print(f'Test   hilang: {len(test_missing)}')\n\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "categories = ['Train - Ada', 'Train - Hilang', 'Test - Ada', 'Test - Hilang']\n"
    "counts = [\n"
    "    len(train_df) - len(train_missing), len(train_missing),\n"
    "    len(test_df)  - len(test_missing),  len(test_missing),\n"
    "]\n"
    "colors = sns.color_palette(CFG.PALETTE, n_colors=4)\n"
    "bars = ax.bar(categories, counts, color=colors)\n"
    "for bar, val in zip(bars, counts):\n"
    "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,\n"
    "            str(val), ha='center', va='bottom')\n"
    "ax.set(title='Ketersediaan File Gambar', ylabel='Jumlah')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_01_file_hilang.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "# Hapus baris yang file-nya tidak ada\n"
    "if train_missing:\n"
    "    train_df = train_df[~train_df['Image'].isin(train_missing)].reset_index(drop=True)\n"
    "    print(f'[OK] {len(train_missing)} baris dihapus dari train_df')\n"
    "if test_missing:\n"
    "    test_df = test_df[~test_df['Image'].isin(test_missing)].reset_index(drop=True)\n"
    "    print(f'[OK] {len(test_missing)} baris dihapus dari test_df')\n"
    "if not train_missing and not test_missing:\n"
    "    print('[OK] Semua file tersedia')"
))
cells.append(md(
    "> **Insight:** Seluruh 12.500 gambar *train* dan 12.500 gambar *test* tersedia "
    "di disk tanpa satu *file* pun yang hilang. *Dataset* dapat langsung diproses "
    "tanpa perlu menghapus baris dari *dataframe*."
))

cells.append(md(
    "## Analisis 2: Pengecekan Duplikat\n\n"
    "> Apakah terdapat gambar duplikat di dalam *training set* maupun antara *training set* dan *test set*?"
))
cells.append(md(
    "Setiap gambar di-*hash* menggunakan SHA-256 untuk menghasilkan sidik jari unik berbasis konten. "
    "Dua gambar dengan nilai *hash* yang sama dipastikan identik secara piksel, terlepas dari nama file. "
    "Pemeriksaan dilakukan pada dua arah, yaitu duplikat di dalam *training set* (train-train) "
    "dan gambar yang muncul di *training set* sekaligus di *test set* (train-test)."
))
cells.append(code(
    "def sha256_file(path: Path) -> str:\n"
    "    h = hashlib.sha256()\n"
    "    with open(path, 'rb') as f:\n"
    "        for chunk in iter(lambda: f.read(65536), b''):\n"
    "            h.update(chunk)\n"
    "    return h.hexdigest()\n\n"
    "print('Menghitung hash SHA-256 untuk seluruh gambar...')\n"
    "train_df['sha256'] = [\n"
    "    sha256_file(CFG.TRAIN_IMG_DIR / fid)\n"
    "    for fid in tqdm(train_df['Image'], desc='Train')\n"
    "]\n"
    "test_df['sha256'] = [\n"
    "    sha256_file(CFG.TEST_IMG_DIR / fid)\n"
    "    for fid in tqdm(test_df['Image'], desc='Test')\n"
    "]\n\n"
    "# Train-train duplicates\n"
    "train_hash_counts = train_df['sha256'].value_counts()\n"
    "dup_hashes_tt     = train_hash_counts[train_hash_counts > 1]\n"
    "dup_train         = train_df[train_df['sha256'].isin(dup_hashes_tt.index)]\n\n"
    "# Train-test duplicates\n"
    "test_hashes  = set(test_df['sha256'])\n"
    "dup_train_test = train_df[train_df['sha256'].isin(test_hashes)]\n\n"
    "print(f'Duplikat train-train : {len(dup_train)} gambar ({len(dup_hashes_tt)} hash unik)')\n"
    "print(f'Duplikat train-test  : {len(dup_train_test)} gambar dari train muncul di test')\n\n"
    "if not dup_train.empty:\n"
    "    print('\\nContoh duplikat train-train:')\n"
    "    display(dup_train.sort_values('sha256').head(10))\n"
    "if not dup_train_test.empty:\n"
    "    print('\\nContoh duplikat train-test:')\n"
    "    display(dup_train_test.head(10))"
))
cells.append(md(
    "> **Insight:** Ditemukan 4 duplikat internal *train* (2 pasang *hash* identik) "
    "dan 4 gambar *train* yang juga muncul di *test set*. Jumlahnya sangat kecil "
    "(0,03% dari *training set*) sehingga tidak berdampak pada pelatihan. "
    "Duplikat *train-test* tidak dihapus karena penghapusan dapat mengganggu "
    "konsistensi evaluasi."
))

cells.append(md("## Analisis 3: Distribusi Kelas\n\n> Apakah *dataset* seimbang antara kelas kucing dan anjing?"))
cells.append(code(
    "label_counts = train_df[CFG.TARGET_COL].value_counts().sort_index()\n"
    "class_labels = [CFG.CLASS_NAMES[i] for i in label_counts.index]\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n\n"
    "# Bar chart\n"
    "sns.barplot(x=class_labels, y=label_counts.values,\n"
    "            palette=CFG.PALETTE, ax=axes[0])\n"
    "for bar, val in zip(axes[0].patches, label_counts.values):\n"
    "    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,\n"
    "                 str(val), ha='center', va='bottom', fontsize=11)\n"
    "axes[0].set(title='Jumlah Sampel per Kelas', xlabel='Kelas', ylabel='Jumlah')\n\n"
    "# Pie chart\n"
    "axes[1].pie(label_counts.values, labels=class_labels, autopct='%1.1f%%',\n"
    "            colors=sns.color_palette(CFG.PALETTE, n_colors=len(class_labels)),\n"
    "            startangle=90)\n"
    "axes[1].set_title('Proporsi Kelas')\n\n"
    "plt.suptitle('Distribusi Kelas - Training Set', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_03_distribusi_kelas.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print(label_counts.to_string())\n"
    "ratio = ' : '.join(f'{CFG.CLASS_NAMES[i]}={label_counts.get(i, 0)}' for i in range(CFG.N_CLASSES))\n"
    "print(f'Rasio kelas : {ratio}')"
))
cells.append(md(
    "> **Insight:** *Dataset* hampir seimbang sempurna: cat 6.192 sampel (49,5%) "
    "dan dog 6.308 sampel (50,5%) dengan selisih hanya 116 sampel. Ketidakseimbangan "
    "yang tidak signifikan ini membuat *class weighting* atau *oversampling* tidak "
    "diperlukan sehingga *macro F1* dapat digunakan sebagai metrik tanpa koreksi apapun."
))

cells.append(md("## Analisis 4: Distribusi Ukuran Gambar\n\n> Apakah dimensi gambar konsisten di seluruh *dataset*?"))
cells.append(code(
    "def collect_sizes(df, images_dir):\n"
    "    widths, heights = [], []\n"
    "    for fid in tqdm(df['Image'], desc='Reading sizes', leave=True):\n"
    "        with Image.open(images_dir / fid) as img:\n"
    "            w, h = img.size\n"
    "        widths.append(w)\n"
    "        heights.append(h)\n"
    "    return widths, heights\n\n"
    "train_w, train_h = collect_sizes(train_df, CFG.TRAIN_IMG_DIR)\n"
    "train_df['width']  = train_w\n"
    "train_df['height'] = train_h\n"
    "train_df['aspect'] = train_df['width'] / train_df['height']\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n\n"
    "# Width distribution\n"
    "sns.histplot(train_df['width'], bins=40, kde=True,\n"
    "             color=sns.color_palette(CFG.PALETTE)[0], ax=axes[0])\n"
    "axes[0].set(title='Distribusi Lebar Gambar', xlabel='Width (px)', ylabel='Count')\n\n"
    "# Height distribution\n"
    "sns.histplot(train_df['height'], bins=40, kde=True,\n"
    "             color=sns.color_palette(CFG.PALETTE)[2], ax=axes[1])\n"
    "axes[1].set(title='Distribusi Tinggi Gambar', xlabel='Height (px)', ylabel='Count')\n\n"
    "# Width vs Height scatter\n"
    "sns.scatterplot(data=train_df, x='width', y='height',\n"
    "                hue=CFG.TARGET_COL, palette=CFG.PALETTE,\n"
    "                alpha=0.4, s=15, ax=axes[2])\n"
    "axes[2].set(title='Width vs Height', xlabel='Width (px)', ylabel='Height (px)')\n"
    "handles, _ = axes[2].get_legend_handles_labels()\n"
    "axes[2].legend(handles, CFG.CLASS_NAMES, title='Kelas')\n\n"
    "plt.suptitle('Distribusi Ukuran Gambar - Training Set', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_04_distribusi_ukuran.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print(train_df[['width', 'height', 'aspect']].describe().round(1))\n"
    "n_unique = train_df.groupby(['width', 'height']).ngroups\n"
    "print(f'Kombinasi ukuran unik: {n_unique}')"
))
cells.append(md(
    "> **Insight:** Dimensi gambar sangat heterogen dengan 5.017 kombinasi unik. "
    "Lebar berkisar 42 hingga 1.050 px dan tinggi 33 hingga 702 px sehingga *resize* "
    "ke dimensi tetap wajib dilakukan. Nilai rata-rata (404 x 361 px) sudah mendekati "
    "target 224 x 224 sehingga *downscaling* tidak menyebabkan kehilangan detail "
    "fitur yang berarti."
))

cells.append(md(
    "## Analisis 5: Pemeriksaan Mode Gambar\n\n"
    "> Apakah semua gambar bertipe RGB, atau terdapat mode lain seperti grayscale atau RGBA?"
))
cells.append(md(
    "Mode gambar menentukan jumlah channel yang diterima model. "
    "Gambar grayscale (`L`) hanya memiliki 1 channel, sementara `RGBA` memiliki 4 channel, "
    "keduanya tidak kompatibel dengan bobot pretrained yang mengharapkan 3 channel RGB. "
    "Hasil analisis ini menentukan apakah konversi `.convert('RGB')` wajib diterapkan "
    "di dalam `Dataset`."
))
cells.append(code(
    "modes = []\n"
    "for fid in tqdm(train_df['Image'], desc='Mode check', leave=True):\n"
    "    with Image.open(CFG.TRAIN_IMG_DIR / fid) as img:\n"
    "        modes.append(img.mode)\n"
    "train_df['img_mode'] = modes\n"
    "mode_counts = Counter(modes)\n\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "sns.barplot(x=list(mode_counts.keys()), y=list(mode_counts.values()),\n"
    "            palette=CFG.PALETTE, ax=ax)\n"
    "for bar, val in zip(ax.patches, mode_counts.values()):\n"
    "    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,\n"
    "            str(val), ha='center', va='bottom')\n"
    "ax.set(title='Distribusi Mode Gambar - Training Set', xlabel='Mode', ylabel='Jumlah')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_05_mode_gambar.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "non_rgb = {k: v for k, v in mode_counts.items() if k != 'RGB'}\n"
    "print(f'Mode counts : {dict(mode_counts)}')\n"
    "print(f'Non-RGB     : {non_rgb if non_rgb else \"tidak ada\"}')\n"
    "print(f'.convert(\"RGB\") : {\"wajib\" if non_rgb else \"opsional (semua sudah RGB)\"}')"
))
cells.append(md(
    "> **Insight:** Seluruh 12.500 gambar *training* berstatus RGB. "
    "Pemanggilan `.convert('RGB')` di dalam *Dataset* tetap dipertahankan sebagai "
    "pelindung untuk data di luar *training set* yang mungkin memiliki mode berbeda "
    "(*grayscale*, RGBA), meskipun untuk data ini tidak wajib."
))

cells.append(md(
    "## Analisis 6: Distribusi Rasio Aspek\n\n"
    "> Seberapa beragam proporsi dimensi gambar, dan apakah banyak yang jauh dari persegi?"
))
cells.append(md(
    "Rasio aspek (lebar/tinggi) menentukan seberapa banyak informasi yang hilang saat "
    "gambar diubah ukurannya menjadi persegi untuk input model. "
    "Gambar dengan rasio jauh dari 1.0 akan mengalami distorsi signifikan jika hanya "
    "di-*resize* tanpa *crop*, sehingga strategi `Resize` diikuti `CenterCrop` atau "
    "`RandomCrop` perlu dipertimbangkan."
))
cells.append(code(
    "# Kolom aspect sudah dihitung di Analisis 3\n"
    "def aspect_cat(r):\n"
    "    if r < 0.9:   return 'Portrait (<0.9)'\n"
    "    elif r <= 1.1: return 'Square (0.9-1.1)'\n"
    "    else:          return 'Landscape (>1.1)'\n\n"
    "train_df['aspect_cat'] = train_df['aspect'].apply(aspect_cat)\n"
    "cat_counts = train_df['aspect_cat'].value_counts()\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n\n"
    "sns.histplot(train_df['aspect'], bins=50, kde=True,\n"
    "             color=sns.color_palette(CFG.PALETTE)[1], ax=axes[0])\n"
    "axes[0].axvline(1.0, color='red', linestyle='--', linewidth=1.2, label='Square (1.0)')\n"
    "axes[0].set(title='Distribusi Rasio Aspek', xlabel='Width / Height', ylabel='Count')\n"
    "axes[0].legend()\n\n"
    "sns.barplot(x=cat_counts.index, y=cat_counts.values,\n"
    "            palette=CFG.PALETTE, ax=axes[1])\n"
    "for bar, val in zip(axes[1].patches, cat_counts.values):\n"
    "    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,\n"
    "                 str(val), ha='center', va='bottom')\n"
    "axes[1].set(title='Kategori Rasio Aspek', xlabel='Kategori', ylabel='Jumlah')\n\n"
    "plt.suptitle('Distribusi Rasio Aspek - Training Set', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_06_rasio_aspek.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print(train_df['aspect'].describe().round(3))"
))
cells.append(md(
    "> **Insight:** Sebagian besar gambar (60,2%) berbentuk *landscape*, hanya 16,7% "
    "yang mendekati persegi. Rasio aspek minimum 0,307 dan maksimum 5,909 menunjukkan "
    "variasi ekstrem. `Resize` polos ke 224x224 akan mendistorsi gambar secara "
    "signifikan; strategi `RandomResizedCrop` dipilih agar model terlatih lebih "
    "*robust* terhadap "
    "berbagai proporsi dimensi tanpa distorsi berlebihan."
))

cells.append(md(
    "## Analisis 7: Distribusi Kecerahan dan Kontras\n\n"
    "> Seberapa bervariasi tingkat kecerahan dan kontras gambar di dalam *training set*?"
))
cells.append(md(
    "Kecerahan diukur sebagai rata-rata piksel ternormalisasi ([0, 1]) dan kontras "
    "sebagai standar deviasi piksel. Distribusi yang lebar pada salah satu dimensi "
    "menunjukkan bahwa augmentasi `ColorJitter(brightness=..., contrast=...)` akan "
    "membantu model belajar fitur yang lebih robust terhadap kondisi pencahayaan. "
    "Distribusi yang sempit sebaliknya mengindikasikan bahwa augmentasi tersebut "
    "kurang diperlukan dan berisiko menambah *noise*."
))
cells.append(code(
    "brightnesses, contrasts = [], []\n"
    "for fid in tqdm(train_df['Image'], desc='Brightness/Contrast', leave=True):\n"
    "    with Image.open(CFG.TRAIN_IMG_DIR / fid) as img:\n"
    "        arr = np.array(img.convert('RGB'), dtype=np.float32) / 255.0\n"
    "    brightnesses.append(float(arr.mean()))\n"
    "    contrasts.append(float(arr.std()))\n\n"
    "train_df['brightness'] = brightnesses\n"
    "train_df['contrast']   = contrasts\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n\n"
    "sns.histplot(train_df['brightness'], bins=40, kde=True,\n"
    "             color=sns.color_palette(CFG.PALETTE)[0], ax=axes[0])\n"
    "axes[0].set(title='Distribusi Kecerahan (Mean Pixel)',\n"
    "            xlabel='Brightness [0-1]', ylabel='Count')\n\n"
    "sns.histplot(train_df['contrast'], bins=40, kde=True,\n"
    "             color=sns.color_palette(CFG.PALETTE)[3], ax=axes[1])\n"
    "axes[1].set(title='Distribusi Kontras (Std Pixel)',\n"
    "            xlabel='Contrast (Std) [0-1]', ylabel='Count')\n\n"
    "plt.suptitle('Kecerahan dan Kontras - Training Set', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_07_kecerahan_kontras.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()\n\n"
    "print(train_df[['brightness', 'contrast']].describe().round(3))"
))
cells.append(md(
    "> **Insight:** Kecerahan memiliki sebaran lebar (min=0,083, max=0,967, "
    "std=0,108) dengan distribusi *bell curve* terpusat di 0,45. Kontras juga "
    "bervariasi (min=0,078, max=0,451) dengan *mean* 0,23. Nilai minimum kecerahan "
    "yang mencapai 0,083 memotivasi `ColorJitter(brightness=0.4)` agar model "
    "terlatih mengenali subjek dalam kondisi pencahayaan sangat gelap hingga terang."
))

cells.append(md(
    "## Analisis 8: Sampel Gambar per Kelas\n\n"
    "> Seperti apa gambar aktual dalam *dataset* untuk setiap kelas?"
))
cells.append(md(
    "Inspeksi visual mengungkap hal-hal yang tidak tertangkap oleh statistik: "
    "keberadaan *watermark*, border putih/hitam, gambar berkualitas rendah, sudut "
    "pengambilan tidak lazim, atau oklusi subjek. "
    "Temuan ini menentukan apakah augmentasi seperti `RandomRotation`, "
    "`RandomPerspective`, atau `GaussianBlur` relevan diterapkan."
))
cells.append(code(
    "n_per_class = 5\n"
    "fig, axes = plt.subplots(len(CFG.CLASS_NAMES), n_per_class,\n"
    "                         figsize=(3 * n_per_class, 3 * len(CFG.CLASS_NAMES)))\n\n"
    "for row, (cls_idx, cls_name) in enumerate(enumerate(CFG.CLASS_NAMES)):\n"
    "    samples = (\n"
    "        train_df[train_df[CFG.TARGET_COL] == cls_idx]\n"
    "        .sample(n_per_class, random_state=CFG.SEED)\n"
    "        .reset_index(drop=True)\n"
    "    )\n"
    "    for col in range(n_per_class):\n"
    "        img = Image.open(CFG.TRAIN_IMG_DIR / samples.loc[col, 'Image'])\n"
    "        axes[row, col].imshow(img)\n"
    "        axes[row, col].set_title(\n"
    "            f'Label: {cls_name}\\n{samples.loc[col, \"Image\"]}',\n"
    "            fontsize=7\n"
    "        )\n"
    "        axes[row, col].axis('off')\n\n"
    "plt.suptitle('Sampel Gambar per Kelas (5 per kelas)', fontsize=13, fontweight='bold')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'eda_08_sampel_gambar.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))
cells.append(md(
    "> **Insight:** Inspeksi visual menunjukkan variasi komposisi tinggi: subjek "
    "tidak selalu terpusat, latar belakang kompleks, dan beberapa gambar memiliki "
    "pencahayaan rendah. Tidak ditemukan *watermark* sistematis. Temuan ini "
    "mengonfirmasi perlunya `RandomHorizontalFlip` dan `RandomResizedCrop` untuk "
    "meningkatkan ketahanan model "
    "terhadap variasi *framing* dan orientasi subjek."
))

# 2. Pembagian Data
cells.append(section("Pembagian Data Latih dan Validasi", "2"))

cells.append(md(
    "Pembagian dilakukan **sebelum** prapemrosesan untuk mencegah kebocoran data "
    "(*data leakage*). Data latih asli sebelum pembagian disimpan karena akan digunakan "
    "kembali saat melatih model final untuk submisi Kaggle."
))

cells.append(code(
    "train_original = train_df.copy()  # simpan untuk pelatihan ulang saat submisi\n\n"
    "train_set, val_set = train_test_split(\n"
    "    train_df,\n"
    "    test_size=CFG.VAL_SPLIT,\n"
    "    random_state=CFG.SEED,\n"
    "    stratify=train_df[CFG.TARGET_COL],\n"
    ")\n"
    "train_set = train_set.reset_index(drop=True)\n"
    "val_set   = val_set.reset_index(drop=True)\n\n"
    "print(f'Latih     : {len(train_set)} sampel')\n"
    "print(f'Validasi  : {len(val_set)} sampel')\n"
    "print(f'Distribusi kelas (latih):\\n{train_set[CFG.TARGET_COL].value_counts()}')"
))

# 3. Pembersihan dan Prapemrosesan Data
cells.append(section("Pembersihan dan Prapemrosesan Data", "3"))

cells.append(md(
    "Data mentah jarang siap digunakan langsung untuk pelatihan. Setiap langkah "
    "prapemrosesan yang dilakukan disertai penjelasan alasan pemilihannya pada sel "
    "*markdown* di bawah kode. Untuk pemrosesan gambar, dapat digunakan OpenCV, "
    "PIL, *torchvision*, Keras, atau pustaka lain yang sesuai."
))

cells.append(md("## Langkah Prapemrosesan 1"))
cells.append(code(
    "# Tulis kode di sini\n"
))
cells.append(md("**Alasan:** ..."))

cells.append(md("## Langkah Prapemrosesan 2"))
cells.append(code(
    "# Tulis kode di sini\n"
))
cells.append(md("**Alasan:** ..."))

cells.append(md("## Kompilasi Pipeline Prapemrosesan"))
cells.append(md(
    "Transformasi akhir didefinisikan untuk data latih dan validasi. "
    "`RandomResizedCrop` menangani rentang rasio aspek lebar (0,3-5,9) yang "
    "teridentifikasi dalam EDA tanpa mendistorsi gambar seperti halnya `Resize(224, 224)` "
    "langsung. `ColorJitter` merespons sebaran kecerahan lebar (min=0,083) dari EDA. "
    "Transformasi validasi menggunakan `Resize(256)+CenterCrop(224)`, protokol evaluasi "
    "ImageNet standar yang sesuai dengan cara bobot *pretrained* divalidasi aslinya."
))
cells.append(code(
    "train_transform = transforms.Compose([\n"
    "    transforms.RandomResizedCrop(CFG.IMG_SIZE[0], scale=(0.6, 1.0)),\n"
    "    transforms.RandomHorizontalFlip(),\n"
    "    transforms.ColorJitter(brightness=0.4, contrast=0.2),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(mean=CFG.IMAGENET_MEAN, std=CFG.IMAGENET_STD),\n"
    "])\n\n"
    "val_transform = transforms.Compose([\n"
    "    transforms.Resize(256),\n"
    "    transforms.CenterCrop(CFG.IMG_SIZE[0]),\n"
    "    transforms.ToTensor(),\n"
    "    transforms.Normalize(mean=CFG.IMAGENET_MEAN, std=CFG.IMAGENET_STD),\n"
    "])"
))

cells.append(md("## Dataset dan DataLoader"))
cells.append(code(
    "class PetDataset(Dataset):\n"
    "    def __init__(self, df, images_dir, transform=None, has_label=True):\n"
    "        self.df          = df.reset_index(drop=True)\n"
    "        self.images_dir  = Path(images_dir)\n"
    "        self.transform   = transform\n"
    "        self.has_label   = has_label\n\n"
    "    def __len__(self):\n"
    "        return len(self.df)\n\n"
    "    def __getitem__(self, idx):\n"
    "        row = self.df.iloc[idx]\n"
    "        img = Image.open(self.images_dir / row['Image']).convert('RGB')\n"
    "        if self.transform:\n"
    "            img = self.transform(img)\n"
    "        if self.has_label:\n"
    "            return img, int(row[CFG.TARGET_COL])\n"
    "        return img\n\n\n"
    "train_ds = PetDataset(train_set, CFG.TRAIN_IMG_DIR, transform=train_transform)\n"
    "val_ds   = PetDataset(val_set,   CFG.TRAIN_IMG_DIR, transform=val_transform)\n"
    "test_ds  = PetDataset(test_df,   CFG.TEST_IMG_DIR,  transform=val_transform, has_label=False)\n\n"
    "train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,\n"
    "                          num_workers=CFG.NUM_WORKERS, pin_memory=True)\n"
    "val_loader   = DataLoader(val_ds,   batch_size=CFG.BATCH_SIZE, shuffle=False,\n"
    "                          num_workers=CFG.NUM_WORKERS, pin_memory=True)\n"
    "test_loader  = DataLoader(test_ds,  batch_size=CFG.BATCH_SIZE, shuffle=False,\n"
    "                          num_workers=CFG.NUM_WORKERS, pin_memory=True)\n\n"
    "print(f'Batch latih    : {len(train_loader)}')\n"
    "print(f'Batch validasi : {len(val_loader)}')"
))
cells.append(md(
    "> **Insight Prapemrosesan:** Dua langkah utama diterapkan: konversi mode gambar "
    "ke RGB dan normalisasi menggunakan statistik ImageNet (nilai rata-rata dan "
    "standar deviasi). `RandomResizedCrop` dipilih berdasarkan temuan EDA rasio "
    "aspek 0,3-5,9 untuk menghindari distorsi; `ColorJitter(brightness=0.4)` "
    "merespons sebaran kecerahan lebar (min=0,083). Transformasi validasi "
    "menggunakan `Resize(256)+CenterCrop(224)` mengikuti protokol evaluasi "
    "ImageNet standar sehingga kondisi *inferensi* konsisten dengan cara bobot "
    "*pretrained* divalidasi."
))

# 4. Pemodelan dan Validasi
cells.append(section("Pemodelan dan Validasi", "4"))

cells.append(md(
    "Dua model dibangun dan dibandingkan, yaitu AlexNet yang diimplementasikan dari nol "
    "menggunakan PyTorch serta model *pretrained* berbasis CNN yang di-*fine-tune* "
    "pada *dataset* ini. Metrik validasi yang digunakan adalah **macro F1-score**."
))

# 4.1 CNN
cells.append(md("## 4.1 Convolutional Neural Network"))

# Utilitas pelatihan
cells.append(md("### Training Utilities"))
cells.append(md(
    "`train_one_epoch` dan `evaluate` menangani satu iterasi latih/validasi dengan "
    "dukungan *mixed precision* (AMP). `run_training` mengelola *loop* pelatihan penuh "
    "dengan *early stopping* untuk AlexNet. `run_finetuning` mengimplementasikan "
    "strategi dua fase: Fase 1 (*warm-up*) membekukan seluruh *backbone* hingga "
    "*val_loss* melandai selama `WARMUP_PATIENCE` *epoch*, lalu Fase 2 men-*unfreeze* "
    "blok terakhir dengan LR diferensial dan *cosine annealing*. "
    "`_unfreeze_last_blocks` membedakan model dengan atribut `features` (MobileNet) "
    "dari keluarga ResNet/ResNeXt menggunakan `named_children()` sehingga "
    "*unfreezing* berlaku pada blok arsitektur yang tepat."
))
cells.append(code(
    "def train_one_epoch(model, loader, optimizer, criterion, scaler):\n"
    "    model.train()\n"
    "    total_loss, preds_all, labels_all = 0.0, [], []\n"
    "    for imgs, labels in tqdm(loader, leave=True):\n"
    "        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n"
    "        optimizer.zero_grad()\n"
    "        with torch.autocast(device_type='cuda', enabled=CFG.USE_AMP):\n"
    "            logits = model(imgs)\n"
    "            loss   = criterion(logits, labels)\n"
    "        scaler.scale(loss).backward()\n"
    "        scaler.step(optimizer)\n"
    "        scaler.update()\n"
    "        total_loss  += loss.item() * len(labels)\n"
    "        preds_all   += logits.argmax(1).cpu().tolist()\n"
    "        labels_all  += labels.cpu().tolist()\n"
    "    avg_loss = total_loss / len(loader.dataset)\n"
    "    f1       = f1_score(labels_all, preds_all, average='macro')\n"
    "    return avg_loss, f1\n\n\n"
    "@torch.no_grad()\n"
    "def evaluate(model, loader, criterion):\n"
    "    model.eval()\n"
    "    total_loss, preds_all, labels_all = 0.0, [], []\n"
    "    for imgs, labels in loader:\n"
    "        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)\n"
    "        logits = model(imgs)\n"
    "        loss   = criterion(logits, labels)\n"
    "        total_loss  += loss.item() * len(labels)\n"
    "        preds_all   += logits.argmax(1).cpu().tolist()\n"
    "        labels_all  += labels.cpu().tolist()\n"
    "    avg_loss = total_loss / len(loader.dataset)\n"
    "    f1       = f1_score(labels_all, preds_all, average='macro')\n"
    "    return avg_loss, f1, preds_all, labels_all\n\n\n"
    "def run_training(model, train_loader, val_loader, checkpoint_path,\n"
    "                 epochs=CFG.EPOCHS, lr=CFG.LR_ALEX,\n"
    "                 early_stop_patience=CFG.EARLY_STOP_PATIENCE):\n"
    "    model.to(DEVICE)\n"
    "    optimizer = torch.optim.Adam(model.parameters(), lr=lr,\n"
    "                                 weight_decay=CFG.WEIGHT_DECAY)\n"
    "    criterion = nn.CrossEntropyLoss()\n"
    "    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)\n"
    "    best_f1   = 0.0\n"
    "    no_improve = 0\n"
    "    history   = []\n"
    "    for epoch in range(1, epochs + 1):\n"
    "        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer,\n"
    "                                          criterion, scaler)\n"
    "        vl_loss, vl_f1, _, _ = evaluate(model, val_loader, criterion)\n"
    "        history.append({'epoch': epoch, 'tr_loss': tr_loss, 'tr_f1': tr_f1,\n"
    "                         'vl_loss': vl_loss, 'vl_f1': vl_f1})\n"
    "        print(f'Epoch {epoch:02d}/{epochs}  '\n"
    "              f'train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  '\n"
    "              f'val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}')\n"
    "        if vl_f1 > best_f1:\n"
    "            best_f1   = vl_f1\n"
    "            no_improve = 0\n"
    "            torch.save(model.state_dict(), checkpoint_path)\n"
    "            print(f'  [OK] Checkpoint disimpan (val_f1={best_f1:.4f})')\n"
    "        else:\n"
    "            no_improve += 1\n"
    "            if no_improve >= early_stop_patience:\n"
    "                print(f'  [Early Stop] Tidak ada perbaikan selama {early_stop_patience} epoch.')\n"
    "                break\n"
    "    print(f'Val macro F1 terbaik: {best_f1:.4f}')\n"
    "    return history\n\n\n"
    "def _split_params(model):\n"
    "    \"\"\"Pisahkan parameter kepala (head) dan backbone berdasarkan nama layer.\"\"\"\n"
    "    head, backbone = [], []\n"
    "    for name, param in model.named_parameters():\n"
    "        if 'classifier' in name or name.startswith('fc.'):\n"
    "            head.append(param)\n"
    "        else:\n"
    "            backbone.append(param)\n"
    "    return head, backbone\n\n\n"
    "def _unfreeze_last_blocks(model, n=2):\n"
    "    if hasattr(model, 'features'):\n"
    "        blocks = list(model.features.children())\n"
    "        for block in blocks[-n:]:\n"
    "            for p in block.parameters():\n"
    "                p.requires_grad = True\n"
    "    else:\n"
    "        named = [(nm, m) for nm, m in model.named_children()\n"
    "                 if nm not in ('fc', 'avgpool')]\n"
    "        for _, module in named[-n:]:\n"
    "            for p in module.parameters():\n"
    "                p.requires_grad = True\n\n\n"
    "def run_finetuning(model, train_loader, val_loader, checkpoint_path,\n"
    "                   epochs=CFG.EPOCHS, lr_head=CFG.LR_HEAD,\n"
    "                   lr_backbone=CFG.LR_BACKBONE,\n"
    "                   warmup_patience=CFG.WARMUP_PATIENCE,\n"
    "                   warmup_min_delta=CFG.WARMUP_MIN_DELTA,\n"
    "                   unfreeze_blocks=CFG.UNFREEZE_BLOCKS,\n"
    "                   early_stop_patience=CFG.EARLY_STOP_PATIENCE,\n"
    "                   cosine_t_max=CFG.COSINE_T_MAX):\n"
    "    model.to(DEVICE)\n"
    "    criterion = nn.CrossEntropyLoss()\n"
    "    scaler    = torch.cuda.amp.GradScaler(enabled=CFG.USE_AMP)\n"
    "    best_f1   = 0.0\n"
    "    history   = []\n\n"
    "    # --- Fase 1: warm-up kepala ---\n"
    "    for p in model.parameters():\n"
    "        p.requires_grad = False\n"
    "    head_params, _ = _split_params(model)\n"
    "    for p in head_params:\n"
    "        p.requires_grad = True\n\n"
    "    optimizer     = torch.optim.Adam(\n"
    "        filter(lambda p: p.requires_grad, model.parameters()),\n"
    "        lr=lr_head, weight_decay=CFG.WEIGHT_DECAY\n"
    "    )\n"
    "    best_val_loss = float('inf')\n"
    "    no_improve    = 0\n"
    "    phase1_end    = epochs   # fallback: tetap di fase 1 jika tidak pernah stabil\n\n"
    "    for epoch in range(1, epochs + 1):\n"
    "        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer,\n"
    "                                          criterion, scaler)\n"
    "        vl_loss, vl_f1, _, _ = evaluate(model, val_loader, criterion)\n"
    "        history.append({'epoch': epoch, 'phase': 1,\n"
    "                         'tr_loss': tr_loss, 'tr_f1': tr_f1,\n"
    "                         'vl_loss': vl_loss, 'vl_f1': vl_f1})\n"
    "        print(f'[Fase 1] Ep {epoch:02d}/{epochs}  '\n"
    "              f'train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  '\n"
    "              f'val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}')\n\n"
    "        if vl_f1 > best_f1:\n"
    "            best_f1 = vl_f1\n"
    "            torch.save(model.state_dict(), checkpoint_path)\n"
    "            print(f'  [OK] Checkpoint disimpan (val_f1={best_f1:.4f})')\n\n"
    "        if best_val_loss - vl_loss > warmup_min_delta:\n"
    "            best_val_loss = vl_loss\n"
    "            no_improve    = 0\n"
    "        else:\n"
    "            no_improve += 1\n\n"
    "        if no_improve >= warmup_patience:\n"
    "            phase1_end = epoch\n"
    "            print(f'[Fase 1] Kepala stabil setelah {epoch} epoch. Beralih ke fine-tune.')\n"
    "            break\n\n"
    "    if phase1_end >= epochs:\n"
    "        print(f'Val macro F1 terbaik: {best_f1:.4f}')\n"
    "        return history\n\n"
    "    # --- Fase 2: unfreeze blok terakhir backbone ---\n"
    "    _unfreeze_last_blocks(model, n=unfreeze_blocks)\n"
    "    head_p, backbone_p = _split_params(model)\n\n"
    "    optimizer = torch.optim.Adam([\n"
    "        {'params': [p for p in backbone_p if p.requires_grad], 'lr': lr_backbone},\n"
    "        {'params': [p for p in head_p    if p.requires_grad], 'lr': lr_head},\n"
    "    ], weight_decay=CFG.WEIGHT_DECAY)\n"
    "    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(\n"
    "        optimizer, T_max=cosine_t_max\n"
    "    )\n"
    "    no_improve = 0\n\n"
    "    for epoch in range(phase1_end + 1, epochs + 1):\n"
    "        tr_loss, tr_f1 = train_one_epoch(model, train_loader, optimizer,\n"
    "                                          criterion, scaler)\n"
    "        vl_loss, vl_f1, _, _ = evaluate(model, val_loader, criterion)\n"
    "        scheduler.step()\n"
    "        history.append({'epoch': epoch, 'phase': 2,\n"
    "                         'tr_loss': tr_loss, 'tr_f1': tr_f1,\n"
    "                         'vl_loss': vl_loss, 'vl_f1': vl_f1})\n"
    "        print(f'[Fase 2] Ep {epoch:02d}/{epochs}  '\n"
    "              f'train_loss={tr_loss:.4f}  train_f1={tr_f1:.4f}  '\n"
    "              f'val_loss={vl_loss:.4f}  val_f1={vl_f1:.4f}')\n\n"
    "        if vl_f1 > best_f1:\n"
    "            best_f1    = vl_f1\n"
    "            no_improve = 0\n"
    "            torch.save(model.state_dict(), checkpoint_path)\n"
    "            print(f'  [OK] Checkpoint disimpan (val_f1={best_f1:.4f})')\n"
    "        else:\n"
    "            no_improve += 1\n"
    "            if no_improve >= early_stop_patience:\n"
    "                print(f'  [Early Stop] Tidak ada perbaikan selama {early_stop_patience} epoch.')\n"
    "                break\n\n"
    "    print(f'Val macro F1 terbaik: {best_f1:.4f}')\n"
    "    return history"
))

# 4.1.0 LeNet
cells.append(md(
    "### 4.1.0 Model LeNet dari Nol\n\n"
    "LeNet-5 adalah arsitektur CNN pelopor yang terdiri dari 2 blok konvolusi-pooling "
    "diikuti 3 lapisan *fully connected*. Setiap blok menggunakan konvolusi 5x5 dan "
    "*average pooling* 2x2 sesuai desain asli LeCun (1998). "
    "`AdaptiveAvgPool2d((5, 5))` digunakan agar peta fitur sebelum lapisan *dense* "
    "tetap berukuran 16x5x5 = 400 unit terlepas dari resolusi masukan 224x224. "
    "Sel ini dinonaktifkan karena kapasitas LeNet jauh di bawah AlexNet untuk gambar "
    "berwarna beresolusi tinggi."
))
cells.append(code(
    "# class LeNet(nn.Module):\n"
    "#     def __init__(self, num_classes: int = 2):\n"
    "#         super().__init__()\n"
    "#         self.features = nn.Sequential(\n"
    "#             # C1\n"
    "#             nn.Conv2d(3, 6, kernel_size=5, padding=0), nn.ReLU(inplace=True),\n"
    "#             # S2\n"
    "#             nn.AvgPool2d(kernel_size=2, stride=2),\n"
    "#             # C3\n"
    "#             nn.Conv2d(6, 16, kernel_size=5, padding=0), nn.ReLU(inplace=True),\n"
    "#             # S4\n"
    "#             nn.AvgPool2d(kernel_size=2, stride=2),\n"
    "#         )\n"
    "#         self.avgpool = nn.AdaptiveAvgPool2d((5, 5))\n"
    "#         self.classifier = nn.Sequential(\n"
    "#             nn.Linear(16 * 5 * 5, 120), nn.ReLU(inplace=True),\n"
    "#             nn.Linear(120, 84),          nn.ReLU(inplace=True),\n"
    "#             nn.Linear(84, num_classes),\n"
    "#         )\n"
    "# \n"
    "#     def forward(self, x):\n"
    "#         x = self.features(x)\n"
    "#         x = self.avgpool(x)\n"
    "#         x = torch.flatten(x, 1)\n"
    "#         return self.classifier(x)\n"
    "# \n"
    "# \n"
    "# lenet = LeNet(num_classes=CFG.N_CLASSES)\n"
    "# n_params = sum(p.numel() for p in lenet.parameters() if p.requires_grad)\n"
    "# print(f'Parameter LeNet: {n_params:,}')"
))

cells.append(md("#### LeNet Training"))
cells.append(code(
    "# seed_everything(CFG.SEED)\n"
    "# lenet         = LeNet(num_classes=CFG.N_CLASSES)\n"
    "# lenet_history = run_training(lenet, train_loader, val_loader,\n"
    "#                              CFG.CHECKPOINT_DIR / 'lenet_best.pt')"
))

cells.append(md("#### Learning Curve - LeNet"))
cells.append(code(
    "# hist_df = pd.DataFrame(lenet_history)\n"
    "# fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "# axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='Train')\n"
    "# axes[0].plot(hist_df['epoch'], hist_df['vl_loss'], label='Validation')\n"
    "# axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss - LeNet')\n"
    "# axes[0].legend()\n"
    "# axes[1].plot(hist_df['epoch'], hist_df['tr_f1'], label='Train')\n"
    "# axes[1].plot(hist_df['epoch'], hist_df['vl_f1'], label='Validation')\n"
    "# axes[1].set(xlabel='Epoch', ylabel='Macro F1', title='Macro F1 - LeNet')\n"
    "# axes[1].legend()\n"
    "# plt.tight_layout()\n"
    "# plt.savefig(FIG_DIR / 'model_lenet_learning_curve.png', dpi=150, bbox_inches='tight')\n"
    "# plt.show()"
))

# 4.1.1 AlexNet
cells.append(md(
    "### 4.1.1 Model AlexNet dari Nol\n\n"
    "AlexNet terdiri dari 5 lapisan konvolusi (dengan ReLU dan *max-pooling*) diikuti "
    "3 lapisan *fully connected*. Lapisan *dense* terakhir disesuaikan menghasilkan "
    "2 logit untuk klasifikasi biner kucing/anjing."
))
cells.append(code(
    "class AlexNet(nn.Module):\n"
    "    def __init__(self, num_classes: int = 2):\n"
    "        super().__init__()\n"
    "        self.features = nn.Sequential(\n"
    "            # Conv1\n"
    "            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0), nn.ReLU(inplace=True),\n"
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n"
    "            # Conv2\n"
    "            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(inplace=True),\n"
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n"
    "            # Conv3\n"
    "            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),\n"
    "            # Conv4\n"
    "            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),\n"
    "            # Conv5\n"
    "            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),\n"
    "            nn.MaxPool2d(kernel_size=3, stride=2),\n"
    "        )\n"
    "        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))\n"
    "        self.classifier = nn.Sequential(\n"
    "            nn.Dropout(0.5),\n"
    "            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),\n"
    "            nn.Dropout(0.5),\n"
    "            nn.Linear(4096, 4096), nn.ReLU(inplace=True),\n"
    "            nn.Linear(4096, num_classes),\n"
    "        )\n\n"
    "    def forward(self, x):\n"
    "        x = self.features(x)\n"
    "        x = self.avgpool(x)\n"
    "        x = torch.flatten(x, 1)\n"
    "        return self.classifier(x)\n\n\n"
    "alexnet = AlexNet(num_classes=CFG.N_CLASSES)\n"
    "n_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)\n"
    "print(f'Parameter AlexNet: {n_params:,}')"
))

cells.append(md("#### AlexNet Training"))
cells.append(code(
    "seed_everything(CFG.SEED)\n"
    "alexnet      = AlexNet(num_classes=CFG.N_CLASSES)\n"
    "alex_history = run_training(alexnet, train_loader, val_loader,\n"
    "                            CFG.CHECKPOINT_ALEX)"
))

cells.append(md("#### Learning Curve - AlexNet"))
cells.append(code(
    "hist_df = pd.DataFrame(alex_history)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "axes[0].plot(hist_df['epoch'], hist_df['tr_loss'], label='Train')\n"
    "axes[0].plot(hist_df['epoch'], hist_df['vl_loss'], label='Validation')\n"
    "axes[0].set(xlabel='Epoch', ylabel='Loss', title='Loss - AlexNet')\n"
    "axes[0].legend()\n"
    "axes[1].plot(hist_df['epoch'], hist_df['tr_f1'], label='Train')\n"
    "axes[1].plot(hist_df['epoch'], hist_df['vl_f1'], label='Validation')\n"
    "axes[1].set(xlabel='Epoch', ylabel='Macro F1', title='Macro F1 - AlexNet')\n"
    "axes[1].legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'model_alexnet_learning_curve.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# 4.1.2 Pretrained fine-tuning
cells.append(md(
    "### 4.1.2 Fine-tuning Model Pretrained\n\n"
    "*Backbone* ResNet-152 *pretrained* pada ImageNet (bobot IMAGENET1K_V2, "
    "akurasi top-1 82,3%) dimuat lalu lapisan keluarannya diganti dengan lapisan "
    "linear untuk klasifikasi biner kucing/anjing."
))
cells.append(code(
    "def build_pretrained(name: str, num_classes: int = 2):\n"
    "    model = tv_models.resnet152(weights=tv_models.ResNet152_Weights.IMAGENET1K_V2)\n"
    "    model.fc = nn.Linear(model.fc.in_features, num_classes)\n"
    "    return model"
))
cells.append(code(
    "pretrained_model = build_pretrained(CFG.PRETRAINED_MODEL, CFG.N_CLASSES)\n"
    "n_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)\n"
    "print(f'Backbone        : {CFG.PRETRAINED_MODEL}')\n"
    "print(f'Parameter terlatih: {n_params:,}')"
))

cells.append(md("#### Fine-tuning Training"))
cells.append(code(
    "seed_everything(CFG.SEED)\n"
    "pretrained_model = build_pretrained(CFG.PRETRAINED_MODEL, CFG.N_CLASSES)\n"
    "ft_history = run_finetuning(pretrained_model, train_loader, val_loader,\n"
    "                            CFG.CHECKPOINT_FT)"
))

cells.append(md("#### Learning Curve - Fine-tuning"))
cells.append(code(
    "hist_df_ft   = pd.DataFrame(ft_history)\n"
    "phase1_mask  = hist_df_ft['phase'] == 1\n"
    "switch_epoch = int(hist_df_ft.loc[phase1_mask, 'epoch'].max()) if phase1_mask.any() else None\n\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
    "for ax, y_tr, y_vl, ylabel, title in [\n"
    "    (axes[0], 'tr_loss', 'vl_loss', 'Loss',     'Loss - Fine-tuning'),\n"
    "    (axes[1], 'tr_f1',  'vl_f1',  'Macro F1', 'Macro F1 - Fine-tuning'),\n"
    "]:\n"
    "    ax.plot(hist_df_ft['epoch'], hist_df_ft[y_tr], label='Train')\n"
    "    ax.plot(hist_df_ft['epoch'], hist_df_ft[y_vl], label='Validation')\n"
    "    if switch_epoch is not None and switch_epoch < hist_df_ft['epoch'].max():\n"
    "        ax.axvline(switch_epoch + 0.5, color='gray', linestyle='--', linewidth=1,\n"
    "                   label=f'Fase 2 mulai (ep {switch_epoch + 1})')\n"
    "    ax.set(xlabel='Epoch', ylabel=ylabel, title=title)\n"
    "    ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'model_finetune_learning_curve.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

# 4.2 Validasi
cells.append(md("## 4.2 Validasi"))
cells.append(md(
    "Kedua model dievaluasi pada *validation set* menggunakan **macro F1-score**. "
    "Hasil yang harus tercantum di *notebook* mencakup hasil validasi kedua model "
    "wajib (AlexNet dan *pretrained*) serta hasil model submisi final di Kaggle."
))
cells.append(code(
    "criterion = nn.CrossEntropyLoss()\n\n"
    "# Muat checkpoint terbaik masing-masing model\n"
    "alexnet.load_state_dict(torch.load(CFG.CHECKPOINT_ALEX, map_location=DEVICE))\n"
    "pretrained_model.load_state_dict(torch.load(CFG.CHECKPOINT_FT, map_location=DEVICE))\n\n"
    "_, alex_f1, alex_preds, alex_labels = evaluate(alexnet, val_loader, criterion)\n"
    "_, ft_f1,   ft_preds,   ft_labels   = evaluate(pretrained_model, val_loader, criterion)\n\n"
    "print(f'AlexNet       val macro F1 : {alex_f1:.4f}')\n"
    "print(f'Fine-tuned    val macro F1 : {ft_f1:.4f}')\n\n"
    "print('\\nClassification Report - AlexNet')\n"
    "print(classification_report(alex_labels, alex_preds,\n"
    "                            target_names=CFG.CLASS_NAMES))\n"
    "print('\\nClassification Report - Fine-tuned')\n"
    "print(classification_report(ft_labels, ft_preds,\n"
    "                            target_names=CFG.CLASS_NAMES))"
))

# Submisi
cells.append(md("## Submisi"))
cells.append(md(
    "Kedua model diprediksi pada *test set* dan masing-masing disimpan ke file CSV terpisah. "
    "Upload file yang sesuai dengan model yang ingin disubmit ke Kaggle."
))
cells.append(code(
    "def predict_test(model, checkpoint_path):\n"
    "    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))\n"
    "    model.to(DEVICE).eval()\n"
    "    preds = []\n"
    "    with torch.no_grad():\n"
    "        for imgs in test_loader:\n"
    "            imgs = imgs.to(DEVICE)\n"
    "            preds.extend(model(imgs).argmax(1).cpu().tolist())\n"
    "    return preds\n\n"
    "# AlexNet submission\n"
    "alex_test_preds = predict_test(alexnet, CFG.CHECKPOINT_ALEX)\n"
    "sub_alex = pd.DataFrame({'Image': test_df['Image'], CFG.TARGET_COL: alex_test_preds})\n"
    "sub_alex.to_csv(CFG.SUBMISSION_ALEX, index=False)\n"
    "print(f'[OK] AlexNet  submisi disimpan di {CFG.SUBMISSION_ALEX}')\n\n"
    "# Fine-tuned submission\n"
    "ft_test_preds = predict_test(pretrained_model, CFG.CHECKPOINT_FT)\n"
    "sub_ft = pd.DataFrame({'Image': test_df['Image'], CFG.TARGET_COL: ft_test_preds})\n"
    "sub_ft.to_csv(CFG.SUBMISSION_FT, index=False)\n"
    "print(f'[OK] Fine-tune submisi disimpan di {CFG.SUBMISSION_FT}')\n\n"
    "print(f'\\nAlexNet  val macro F1 : {alex_f1:.4f}')\n"
    "print(f'Fine-tune val macro F1 : {ft_f1:.4f}')\n"
    "sub_ft.head()"
))
cells.append(md(
    "> **Insight Pemodelan:** AlexNet dari nol mencapai *val macro F1* = 0,9220 "
    "pada *epoch* 25 dengan konvergensi stabil menggunakan LR=1e-4. ResNet-152 "
    "dengan strategi dua fase mencapai *val macro F1* = 0,9960, unggul 7,4 poin "
    "dari AlexNet. Fase 1 (*warm-up* 8 *epoch*) berhasil menstabilkan kepala "
    "klasifikasi sebelum *backbone* *di-unfreeze*; Fase 2 langsung melompat dari "
    "0,9888 ke 0,9928 pada *epoch* pertama sehingga fitur ResNet-152 terbukti "
    "sangat relevan untuk tugas biner ini. *Train F1* Fase 2 mencapai ~0,9998, "
    "menandakan model hampir menghafal data latih, tetapi *val F1* tetap tinggi "
    "berkat augmentasi yang memadai."
))

# 5. Analisis Kesalahan
cells.append(section("Analisis Kesalahan", "5"))

cells.append(md(
    "Kesalahan prediksi model dianalisis untuk memahami penyebab utama "
    "ketidaktepatan klasifikasi. Beberapa pertanyaan panduan yang dapat digunakan:\n\n"
    "1. Bagaimana distribusi kesalahan antarkelas? Apakah sebagian besar kesalahan berasal dari satu kelas?\n"
    "2. Apakah *false positive* atau *false negative* yang lebih dominan?\n"
    "3. Apakah oklusi atau sudut pengambilan gambar yang tidak biasa berkontribusi pada kesalahan?\n"
    "4. Apakah pola tertentu terlihat pada sampel yang salah diklasifikasikan?"
))

cells.append(md("## Confusion Matrix"))
cells.append(code(
    "# Gunakan prediksi dari model terbaik pada validasi\n"
    "best_preds  = ft_preds  if ft_f1 >= alex_f1 else alex_preds\n"
    "best_labels = ft_labels if ft_f1 >= alex_f1 else alex_labels\n\n"
    "cm = confusion_matrix(best_labels, best_preds)\n"
    "fig, ax = plt.subplots(figsize=(5, 4))\n"
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
    "            xticklabels=CFG.CLASS_NAMES, yticklabels=CFG.CLASS_NAMES, ax=ax)\n"
    "ax.set(xlabel='Predicted', ylabel='True Label', title='Confusion Matrix')\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'error_confusion_matrix.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(md("## Misclassified Samples"))
cells.append(code(
    "# Visualisasi gambar yang salah diklasifikasikan\n"
    "wrong_idx = [i for i, (p, t) in enumerate(zip(best_preds, best_labels)) if p != t]\n"
    "print(f'Salah diklasifikasikan: {len(wrong_idx)} / {len(best_labels)}')\n\n"
    "n_show = min(10, len(wrong_idx))\n"
    "n_cols = min(5, n_show)\n"
    "n_rows = (n_show + n_cols - 1) // n_cols\n"
    "fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))\n"
    "axes = np.array(axes).flatten()\n"
    "for ax, idx in zip(axes[:n_show], wrong_idx[:n_show]):\n"
    "    row = val_set.iloc[idx]\n"
    "    img = Image.open(CFG.TRAIN_IMG_DIR / row['Image'])\n"
    "    ax.imshow(img)\n"
    "    ax.set_title(\n"
    "        f'Asli: {CFG.CLASS_NAMES[best_labels[idx]]}\\n'\n"
    "        f'Pred: {CFG.CLASS_NAMES[best_preds[idx]]}',\n"
    "        fontsize=8\n"
    "    )\n"
    "    ax.axis('off')\n"
    "for ax in axes[n_show:]:\n"
    "    ax.set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.savefig(FIG_DIR / 'error_misclassified.png', dpi=150, bbox_inches='tight')\n"
    "plt.show()"
))

cells.append(code(
    "print(f'Daftar sampel salah diklasifikasikan ({len(wrong_idx)} sampel)')\n"
    "for rank, idx in enumerate(wrong_idx, 1):\n"
    "    row = val_set.iloc[idx]\n"
    "    true_label = CFG.CLASS_NAMES[best_labels[idx]]\n"
    "    pred_label = CFG.CLASS_NAMES[best_preds[idx]]\n"
    "    print(f'{rank:>3}. {row[\"Image\"]}  true={true_label:<4}  pred={pred_label}')"
))

cells.append(md(
    "> **Insight Analisis Kesalahan:** Dari 2.500 sampel validasi, hanya 10 yang "
    "salah diklasifikasikan: 6 cat diprediksi dog dan 4 dog diprediksi cat. "
    "Kesalahan terbagi dua kelompok: (1) gambar non-fotografis seperti ilustrasi "
    "sketsa, logo vektor, dan *placeholder* 'No Photo Available', yang tergolong "
    "*domain shift* dan tidak dapat diatasi karena subjek yang dimaksud tidak hadir "
    "dalam gambar; (2) foto *borderline* dengan pose ambigu atau latar belakang "
    "kompleks. Kelompok pertama bersifat *noise* label bawaan *dataset* sehingga "
    "0,9960 merupakan batas praktis yang dapat dicapai dengan *supervised training* "
    "standar."
))

# 6. Insights
cells.append(section("Insights", "6"))

cells.append(md(
    "Insights dari setiap tahap pengerjaan telah disajikan langsung di bawah "
    "masing-masing sel analisis yang sesuai: di bawah setiap analisis EDA, "
    "di akhir seksi Prapemrosesan, di akhir seksi Pemodelan, "
    "dan di akhir seksi Analisis Kesalahan."
))

# Bundling output
cells.append(md(
    "---\n\n"
    "## Bundling Output\n\n"
    "Seluruh gambar yang disimpan selama eksperimen dikemas ke dalam satu file zip "
    "untuk memudahkan pengumpulan dan dokumentasi. Checkpoint model tidak disertakan."
))
cells.append(code(
    "import zipfile as _zf\n\n"
    "bundle_path = CFG.OUTPUT_DIR / 'figures_bundle.zip'\n"
    "figures = sorted(FIG_DIR.glob('*.png'))\n\n"
    "with _zf.ZipFile(bundle_path, 'w', _zf.ZIP_DEFLATED) as zf:\n"
    "    for fig_path in figures:\n"
    "        zf.write(fig_path, arcname=fig_path.name)\n\n"
    "print(f'[OK] {len(figures)} gambar dikemas ke {bundle_path}')\n"
    "for f in figures:\n"
    "    print(f'  {f.name}')"
))

# Tulis notebook
notebook = {
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

OUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"[OK] {len(cells)} sel ditulis -> {OUT}")
