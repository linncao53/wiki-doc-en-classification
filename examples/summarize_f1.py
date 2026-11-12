"""Recompute original-split F1 from the exported per-class scores."""
import csv
from collections import defaultdict
from math import fsum, isfinite
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "results" / "original-split-per-class.csv"


def summarize(path: Path) -> dict[str, tuple[float, float]]:
    scores = defaultdict(dict)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model, label = row["model"], row["class"]
            f1, support = float(row["f1"]), int(row["support"])
            if label in scores[model]:
                raise ValueError(f"Duplicate class {label!r} for {model}")
            if not isfinite(f1) or not 0 <= f1 <= 1 or support < 0:
                raise ValueError(f"Invalid score or support for {model}, {label}")
            scores[model][label] = (f1, support)

    summary = {}
    for model, classes in scores.items():
        total = sum(support for _, support in classes.values())
        if total == 0:
            raise ValueError(f"No test pages for {model}")
        macro = fsum(f1 for f1, _ in classes.values()) / len(classes)
        weighted = fsum(f1 * support for f1, support in classes.values()) / total
        summary[model] = (macro, weighted)
    return summary


if __name__ == "__main__":
    print("Model       Macro F1  Weighted F1")
    for model, (macro, weighted) in summarize(DATA).items():
        print(f"{model:<11} {macro:>8.2%} {weighted:>12.2%}")
