from pathlib import Path

ROOT_A   = Path("dataset")
ROOT_B   = Path("dataset_v2")
SHOW_DIFF = False   # set True to print individual differing filenames

# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANDIDATE_LAYOUTS = [
    ("images/images/train", "images/images/test"),
    ("images/train",        "images/test"),
    ("train",               "test"),
]


def find_split_dirs(root):
    for train_sub, test_sub in CANDIDATE_LAYOUTS:
        t = root / train_sub
        v = root / test_sub
        if t.is_dir() or v.is_dir():
            return t if t.is_dir() else None, v if v.is_dir() else None
    return None, None


def list_images(d):
    if d is None:
        return set()
    return {f.name for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS}


def list_csv(csv_path):
    if not csv_path.exists():
        return None
    import csv
    with open(csv_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        col = next((c for c in (reader.fieldnames or []) if c.strip().lower() == "image"), None)
        if col is None:
            return None
        return {row[col].strip() for row in reader}


def report(label, set_a, set_b, name_a, name_b):
    only_a = set_a - set_b
    only_b = set_b - set_a
    both   = set_a & set_b
    tag    = "[SAME]" if set_a == set_b else "[DIFF]"
    print(f"\n  {tag} {label}")
    print(f"    {name_a:<30} {len(set_a):>6} files")
    print(f"    {name_b:<30} {len(set_b):>6} files")
    print(f"    in both                        {len(both):>6}")
    if set_a != set_b:
        print(f"    only in {name_a:<21} {len(only_a):>6}")
        print(f"    only in {name_b:<21} {len(only_b):>6}")
        if SHOW_DIFF:
            for name, s in ((name_a, only_a), (name_b, only_b)):
                if s:
                    print(f"\n    --- only in {name} ---")
                    for f in sorted(s):
                        print(f"      {f}")

# ---------------------------------------------------------------------------

print(f"A  {ROOT_A.resolve()}")
print(f"B  {ROOT_B.resolve()}")

train_a, test_a = find_split_dirs(ROOT_A)
train_b, test_b = find_split_dirs(ROOT_B)

print("\n=== Image files (filesystem) ===")
if train_a or train_b:
    report("train", list_images(train_a), list_images(train_b), ROOT_A.name, ROOT_B.name)
else:
    print("\n  [SKIP] train -- no train directory found in either root")

if test_a or test_b:
    report("test", list_images(test_a), list_images(test_b), ROOT_A.name, ROOT_B.name)
else:
    print("\n  [SKIP] test  -- no test directory found in either root")

csv_checks = [
    ("train.csv", ROOT_A / "train.csv", ROOT_B / "train.csv"),
    ("test.csv",  ROOT_A / "test.csv",  ROOT_B / "test.csv"),
]

if any(pa.exists() or pb.exists() for _, pa, pb in csv_checks):
    print("\n=== CSV (Image column) ===")
    for label, pa, pb in csv_checks:
        ra, rb = list_csv(pa), list_csv(pb)
        if ra is None and rb is None:
            print(f"\n  [SKIP] {label} -- not found or no Image column")
        else:
            report(label, ra or set(), rb or set(), ROOT_A.name, ROOT_B.name)
