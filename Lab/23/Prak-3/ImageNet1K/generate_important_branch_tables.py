import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COUNT_CSV = ROOT / "ImageNet1k_class_counts.csv"
HIERARCHY_CSV = ROOT / "ImageNet1k_class_hierarchy.csv"
OUTPUT_MD = ROOT / "important_subbranch_tables.md"


BRANCHES = [
    {
        "title": "Dog / Canine",
        "match": "> canine > dog >",
        "note": "Domestic dog branch. This is one of the richest and most transferable ImageNet branches.",
    },
    {
        "title": "Cat / Feline",
        "match": "> feline >",
        "note": "Includes domestic cats and larger wild felines under the feline branch.",
    },
    {
        "title": "Bird",
        "match": "> bird >",
        "note": "Broad bird branch covering land birds, aquatic birds, parrots, raptors, and related groups.",
    },
    {
        "title": "Fish / Aquatic Vertebrate",
        "match": "> fish >",
        "note": "Covers bony fish, sharks, rays, and other fish-related branches inside aquatic vertebrates.",
    },
    {
        "title": "Reptile",
        "match": "> reptile >",
        "note": "Includes turtles, lizards, crocodilians, snakes, and dinosaur-related entries.",
    },
    {
        "title": "Amphibian",
        "match": "> amphibian >",
        "note": "Includes salamanders, newts, frogs, and related amphibian classes.",
    },
    {
        "title": "Primate",
        "match": "> primate >",
        "note": "Includes apes, monkeys, and lemur-adjacent primate groups.",
    },
    {
        "title": "Ungulate / Hoofed Mammal",
        "match": "> ungulate >",
        "note": "Includes equines, swine, bovines, sheep-like, camelid, and related hoofed mammals.",
    },
    {
        "title": "Insect",
        "match": "> insect >",
        "note": "Includes beetles, flies, ants, butterflies, dragonflies, and other insect classes.",
    },
    {
        "title": "Crustacean",
        "match": "> crustacean >",
        "note": "Includes crabs, lobsters, crayfish, hermit crab, isopod, and related crustaceans.",
    },
    {
        "title": "Mollusk",
        "match": "> mollusk >",
        "note": "Includes snail, slug, conch, chiton, nautilus, and related mollusk classes.",
    },
    {
        "title": "Vehicle",
        "match": "> vehicle >",
        "note": "Important artifact branch covering wheeled vehicles, vessels, aircraft, spacecraft, and related conveyances.",
    },
    {
        "title": "Device",
        "match": "> device >",
        "note": "Covers instruments, machines, electronics, optical devices, musical instruments, and related device classes.",
    },
    {
        "title": "Container",
        "match": "> container >",
        "note": "Includes bags, baskets, bottles, jars, vessels, tanks, and other container-like artifacts.",
    },
    {
        "title": "Clothing",
        "match": "> clothing >",
        "note": "Includes garments, outerwear, hats, footwear, neckwear, and other clothing-related classes.",
    },
    {
        "title": "Food Fruit",
        "match": "> edible fruit >",
        "note": "Fruit-like food branch under edible fruit.",
    },
    {
        "title": "Prepared Food / Dish",
        "match": "> dish >",
        "note": "Prepared foods and dish-like classes such as soup, sandwich, pizza, burrito, and related foods.",
    },
    {
        "title": "Beverage",
        "match": "> beverage >",
        "note": "Drink categories such as wine, coffee, punch-like drinks, and related beverage classes.",
    },
    {
        "title": "Fungus",
        "match": "> fungus >",
        "note": "Mushroom and fungus-related biological classes.",
    },
    {
        "title": "Geological Formation",
        "match": "> geological formation >",
        "note": "Natural scene / landform branch including cliffs, shores, reefs, valleys, and mountains.",
    },
]


def load_counts(path: Path) -> dict[int, dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for row in rows:
        row["index"] = int(row["index"])
        row["train_count"] = int(row["train_count"])
        row["val_count"] = int(row["val_count"])
        out[row["index"]] = row
    return out


def load_hierarchy(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["index"] = int(row["index"])
        row["hierarchy_depth"] = int(row["hierarchy_depth"])
    return rows


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        safe = [cell.replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(safe) + " |")
    return lines


def main() -> None:
    counts_by_index = load_counts(COUNT_CSV)
    rows = load_hierarchy(HIERARCHY_CSV)
    for row in rows:
        count_row = counts_by_index[row["index"]]
        row["train_count"] = count_row["train_count"]
        row["val_count"] = count_row["val_count"]
        row["wnid"] = count_row["wnid"]
        row["imagenet_label"] = count_row["imagenet_label"]

    branch_results = []
    for branch in BRANCHES:
        matched = [row for row in rows if branch["match"] in row["hierarchy_path"]]
        matched = sorted(matched, key=lambda row: row["index"])
        branch_results.append(
            {
                "title": branch["title"],
                "note": branch["note"],
                "match": branch["match"],
                "rows": matched,
                "class_count": len(matched),
                "train_count": sum(row["train_count"] for row in matched),
                "val_count": sum(row["val_count"] for row in matched),
            }
        )

    md_lines = [
        "# Important Subbranch Tables",
        "",
        "This file lists the ImageNet1k labels that fall under the main important branches discussed in `important_hierarchy_analysis.md`.",
        "",
        "Source:",
        "",
        "- `ImageNet1k_class_counts.csv`",
        "",
        "Notes:",
        "",
        "- Branches are matched by WordNet hierarchy-path substrings.",
        "- Branches may overlap semantically in a few cases.",
        "- Counts are image counts from ImageNet localization train/val solution files.",
        "",
        "## Summary",
        "",
    ]

    md_lines.extend(
        markdown_table(
            ["Branch", "Matched Path Fragment", "Class Count", "Train Count", "Val Count"],
            [
                [
                    item["title"],
                    item["match"],
                    str(item["class_count"]),
                    str(item["train_count"]),
                    str(item["val_count"]),
                ]
                for item in branch_results
            ],
        )
    )

    for item in branch_results:
        md_lines.extend(
            [
                "",
                f"## {item['title']}",
                "",
                item["note"],
                "",
                f"- Class count: `{item['class_count']}`",
                f"- Train image count: `{item['train_count']}`",
                f"- Val image count: `{item['val_count']}`",
                "",
            ]
        )

        if not item["rows"]:
            md_lines.append("No rows matched this branch.")
            continue

        md_lines.extend(
            markdown_table(
                ["Index", "WNID", "Class Name", "Train Count", "Val Count"],
                [
                    [
                        str(row["index"]),
                        row["wnid"],
                        row["imagenet_label"],
                        str(row["train_count"]),
                        str(row["val_count"]),
                    ]
                    for row in item["rows"]
                ],
            )
        )

    OUTPUT_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
