"""Measures how often /tag is right, on photos you labeled yourself.

Usage:
    uv run python -m eval.evaluate

Put photos in eval/images/ and one line per photo in eval/labels.csv:
    filename,category,pattern,formality
    jeans-01.jpg,bottom,solid,casual
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from app.services.embedder import get_embedder
from app.services.tagging import tag

HERE = Path(__file__).parent
ATTRIBUTES = ["category", "pattern", "formality"]


def main() -> None:
    rows = list(csv.DictReader((HERE / "labels.csv").open()))
    if not rows:
        print("eval/labels.csv is empty. Add labeled photos first (see eval/README.md).")
        return

    embedder = get_embedder()
    correct: Counter[str] = Counter()
    total: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        result = tag(Image.open(HERE / "images" / row["filename"]), embedder)
        for attribute in ATTRIBUTES:
            expected = (row.get(attribute) or "").strip()
            if not expected:
                continue
            predicted = getattr(result, attribute).label
            total[attribute] += 1
            correct[attribute] += predicted == expected
            if attribute == "category":
                confusion[expected][predicted] += 1

    lines = ["# Tagging accuracy", "", f"{len(rows)} labeled photos.", "",
             "| Attribute | Accuracy | Photos |", "|---|---|---|"]
    for attribute in ATTRIBUTES:
        if total[attribute]:
            accuracy = correct[attribute] / total[attribute]
            lines.append(f"| {attribute} | {accuracy:.0%} | {total[attribute]} |")

    labels = sorted({*confusion, *(p for c in confusion.values() for p in c)})
    lines += ["", "## Category confusion (rows: real, columns: predicted)", "",
              "| | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for real in labels:
        counts = " | ".join(str(confusion[real][p]) for p in labels)
        lines.append(f"| **{real}** | {counts} |")

    report = "\n".join(lines) + "\n"
    (HERE / "results.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
