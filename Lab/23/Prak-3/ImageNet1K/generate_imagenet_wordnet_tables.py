import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
IMAGENET_DIR = ROOT / "ImageNet-1K"
WORDNET_DIR = ROOT / "WordNet-3-0" / "dict"

CLASS_INDEX_PATH = IMAGENET_DIR / "ImageNet_class_index.json"
TRAIN_SOLUTION_PATH = IMAGENET_DIR / "LOC_train_solution.csv"
VAL_SOLUTION_PATH = IMAGENET_DIR / "LOC_val_solution.csv"
DATA_NOUN_PATH = WORDNET_DIR / "data.noun"

OUTPUT_MD = ROOT / "ImageNet1k_wordnet_tables.md"
OUTPUT_JSON = ROOT / "ImageNet1k_wordnet_tables.json"
OUTPUT_CLASS_CSV = ROOT / "ImageNet1k_class_hierarchy.csv"
OUTPUT_COUNT_CSV = ROOT / "ImageNet1k_class_counts.csv"
OUTPUT_TREE_JSON = ROOT / "ImageNet1k_hierarchy_count_tree.json"
OUTPUT_NODE_COUNT_CSV = ROOT / "ImageNet1k_hierarchy_node_counts.csv"


def load_class_index(path: Path) -> dict[int, tuple[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): (v[0], v[1]) for k, v in raw.items()}


def load_data_noun(path: Path) -> dict[int, str]:
    synsets = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if len(line) < 9 or not line[:8].isdigit():
                continue
            synsets[int(line[:8])] = line.rstrip("\n")
    return synsets


def parse_synset(line: str) -> dict:
    data_part, _, gloss = line.partition("|")
    tokens = data_part.strip().split()

    offset = int(tokens[0])
    lex_filenum = tokens[1]
    ss_type = tokens[2]
    w_cnt = int(tokens[3], 16)

    i = 4
    words = []
    for _ in range(w_cnt):
        words.append(tokens[i].replace("_", " "))
        i += 2

    p_cnt = int(tokens[i])
    i += 1

    pointers = []
    for _ in range(p_cnt):
        pointers.append(
            {
                "symbol": tokens[i],
                "target_offset": int(tokens[i + 1]),
                "pos": tokens[i + 2],
                "source_target": tokens[i + 3],
            }
        )
        i += 4

    return {
        "offset": offset,
        "lex_filenum": lex_filenum,
        "ss_type": ss_type,
        "words": words,
        "pointers": pointers,
        "gloss": gloss.strip(),
    }


def first_hypernym_offset(parsed: dict) -> int | None:
    for pointer in parsed["pointers"]:
        if pointer["symbol"] in {"@", "@i"} and pointer["pos"] == "n":
            return pointer["target_offset"]
    return None


def build_hierarchy(offset: int, synset_map: dict[int, str], cache: dict[int, list[str]]) -> list[str]:
    if offset in cache:
        return cache[offset]

    line = synset_map.get(offset)
    if line is None:
        cache[offset] = []
        return cache[offset]

    parsed = parse_synset(line)
    label = parsed["words"][0] if parsed["words"] else str(offset)
    parent_offset = first_hypernym_offset(parsed)

    if parent_offset is None or parent_offset == offset:
        hierarchy = [label]
    else:
        parent_hierarchy = build_hierarchy(parent_offset, synset_map, cache)
        hierarchy = parent_hierarchy + [label]

    cache[offset] = hierarchy
    return hierarchy


def count_images_per_class(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wnid = row["PredictionString"].split()[0]
            counts[wnid] = counts.get(wnid, 0) + 1
    return counts


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        safe_row = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(safe_row) + " |")
    return lines


def add_to_tree(tree: dict, hierarchy: list[str], train_count: int, val_count: int) -> None:
    node = tree
    for level_name in hierarchy:
        children = node.setdefault("children", {})
        if level_name not in children:
            children[level_name] = {
                "name": level_name,
                "class_count": 0,
                "train_count": 0,
                "val_count": 0,
                "children": {},
            }
        node = children[level_name]
        node["class_count"] += 1
        node["train_count"] += train_count
        node["val_count"] += val_count


def sort_tree(node: dict) -> dict:
    children = node.get("children", {})
    sorted_children = dict(
        sorted(
            ((name, sort_tree(child)) for name, child in children.items()),
            key=lambda item: (-item[1]["train_count"], item[0]),
        )
    )
    node["children"] = sorted_children
    return node


def flatten_tree(node: dict, path: list[str] | None = None, level: int = -1) -> list[dict]:
    if path is None:
        path = []

    rows = []
    if level >= 0:
        rows.append(
            {
                "level": level,
                "node_name": node["name"],
                "parent_name": path[-1] if path else "",
                "class_count": node["class_count"],
                "train_count": node["train_count"],
                "val_count": node["val_count"],
                "path": " > ".join(path + [node["name"]]),
            }
        )

    for child_name, child in node.get("children", {}).items():
        rows.extend(flatten_tree(child, path + ([node["name"]] if level >= 0 else []), level + 1))
    return rows


def main() -> None:
    class_index = load_class_index(CLASS_INDEX_PATH)
    synset_map = load_data_noun(DATA_NOUN_PATH)
    train_counts = count_images_per_class(TRAIN_SOLUTION_PATH)
    val_counts = count_images_per_class(VAL_SOLUTION_PATH)

    hierarchy_cache: dict[int, list[str]] = {}

    class_rows = []
    count_rows = []
    json_records = []
    tree_root = {
        "name": "root",
        "class_count": 0,
        "train_count": 0,
        "val_count": 0,
        "children": {},
    }

    for index in range(1000):
        wnid, imagenet_label = class_index[index]
        noun_offset = int(wnid[1:])
        hierarchy = build_hierarchy(noun_offset, synset_map, hierarchy_cache)
        hierarchy_path = " > ".join(hierarchy)

        class_row = {
            "index": index,
            "wnid": wnid,
            "imagenet_label": imagenet_label,
            "hierarchy_depth": len(hierarchy),
            "hierarchy_path": hierarchy_path,
        }
        count_row = {
            "index": index,
            "wnid": wnid,
            "imagenet_label": imagenet_label,
            "train_count": train_counts.get(wnid, 0),
            "val_count": val_counts.get(wnid, 0),
            "hierarchy_depth": len(hierarchy),
            "hierarchy_path": hierarchy_path,
        }

        class_rows.append(class_row)
        count_rows.append(count_row)
        tree_root["class_count"] += 1
        tree_root["train_count"] += train_counts.get(wnid, 0)
        tree_root["val_count"] += val_counts.get(wnid, 0)
        add_to_tree(
            tree_root,
            hierarchy,
            train_counts.get(wnid, 0),
            val_counts.get(wnid, 0),
        )
        json_records.append(
            {
                "index": index,
                "wnid": wnid,
                "imagenet_label": imagenet_label,
                "hierarchy": hierarchy,
                "train_count": train_counts.get(wnid, 0),
                "val_count": val_counts.get(wnid, 0),
            }
        )

    count_rows_sorted = sorted(
        count_rows,
        key=lambda row: (-int(row["train_count"]), int(row["index"])),
    )
    sorted_tree_root = sort_tree(tree_root)
    node_rows = flatten_tree(sorted_tree_root)
    node_rows = sorted(node_rows, key=lambda row: (int(row["level"]), row["path"]))

    write_csv(
        OUTPUT_CLASS_CSV,
        class_rows,
        ["index", "wnid", "imagenet_label", "hierarchy_depth", "hierarchy_path"],
    )
    write_csv(
        OUTPUT_COUNT_CSV,
        count_rows_sorted,
        ["index", "wnid", "imagenet_label", "train_count", "val_count"],
    )
    write_csv(
        OUTPUT_NODE_COUNT_CSV,
        node_rows,
        ["level", "node_name", "parent_name", "class_count", "train_count", "val_count", "path"],
    )

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "note": (
                    "Hierarchy paths are derived from WordNet 3.0 noun hypernyms using "
                    "the first noun hypernym pointer found in data.noun. ImageNet train/val "
                    "counts come from LOC_train_solution.csv and LOC_val_solution.csv."
                ),
                "source_files": {
                    "class_index": str(CLASS_INDEX_PATH.relative_to(ROOT)),
                    "data_noun": str(DATA_NOUN_PATH.relative_to(ROOT)),
                    "train_solution": str(TRAIN_SOLUTION_PATH.relative_to(ROOT)),
                    "val_solution": str(VAL_SOLUTION_PATH.relative_to(ROOT)),
                },
                "records": json_records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    OUTPUT_TREE_JSON.write_text(
        json.dumps(
            {
                "note": (
                    "Nested hierarchy tree aggregated from ImageNet1k classes using "
                    "WordNet 3.0 noun hypernym paths. Each node contains the number of "
                    "ImageNet classes under that node plus total train/val image counts."
                ),
                "source_files": {
                    "class_index": str(CLASS_INDEX_PATH.relative_to(ROOT)),
                    "data_noun": str(DATA_NOUN_PATH.relative_to(ROOT)),
                    "train_solution": str(TRAIN_SOLUTION_PATH.relative_to(ROOT)),
                    "val_solution": str(VAL_SOLUTION_PATH.relative_to(ROOT)),
                },
                "tree": sorted_tree_root,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_lines = [
        "# ImageNet1k WordNet Tables",
        "",
        "This file contains two tables built from:",
        "",
        "- `ImageNet-1K/ImageNet_class_index.json`",
        "- `WordNet-3-0/dict/data.noun`",
        "- `ImageNet-1K/LOC_train_solution.csv`",
        "- `ImageNet-1K/LOC_val_solution.csv`",
        "",
        "Notes:",
        "",
        "- The hierarchy is based on WordNet 3.0 noun hypernyms.",
        "- Each hierarchy path follows the first available noun hypernym pointer in `data.noun`.",
        "- The count table is sorted by descending `train_count`.",
        "",
        "Related exports:",
        "",
        "- `ImageNet1k_class_hierarchy.csv`",
        "- `ImageNet1k_class_counts.csv`",
        "- `ImageNet1k_hierarchy_node_counts.csv`",
        "- `ImageNet1k_wordnet_tables.json`",
        "- `ImageNet1k_hierarchy_count_tree.json`",
        "",
        "## Table 1: Each Class With Its Hierarchy",
        "",
    ]

    md_lines.extend(
        markdown_table(
            ["Index", "WNID", "Class Name", "Hierarchy Depth", "Hierarchy Path"],
            [
                [
                    str(row["index"]),
                    row["wnid"],
                    row["imagenet_label"],
                    str(row["hierarchy_depth"]),
                    row["hierarchy_path"],
                ]
                for row in class_rows
            ],
        )
    )

    md_lines.extend(
        [
            "",
            "## Table 2: Each Hierarchy Node With Aggregated Counts",
            "",
        ]
    )

    md_lines.extend(
        markdown_table(
            ["Level", "Node Name", "Parent Name", "Class Count", "Train Count", "Val Count", "Path"],
            [
                [
                    str(row["level"]),
                    row["node_name"],
                    row["parent_name"],
                    str(row["class_count"]),
                    str(row["train_count"]),
                    str(row["val_count"]),
                    row["path"],
                ]
                for row in node_rows
            ],
        )
    )

    OUTPUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_CLASS_CSV}")
    print(f"Wrote {OUTPUT_COUNT_CSV}")
    print(f"Wrote {OUTPUT_TREE_JSON}")
    print(f"Wrote {OUTPUT_NODE_COUNT_CSV}")


if __name__ == "__main__":
    main()
