"""Rank original-split classes by the F1 gap between two baselines."""
import argparse
import csv
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "results" / "original-split-per-class.csv"
MODELS = ("DistilBERT", "DiT", "LayoutXLM")


def compare_classes(path, first, second, min_support):
    scores = {first: {}, second: {}}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in scores:
                scores[row["model"]][row["class"]] = (
                    float(row["f1"]), int(row["support"])
                )

    if not scores[first] or scores[first].keys() != scores[second].keys():
        raise ValueError("The two models must report the same nonempty set of classes.")

    comparison = []
    for label, (left, support) in scores[first].items():
        right, other_support = scores[second][label]
        if support != other_support:
            raise ValueError(f"Test supports differ for {label!r}.")
        if support >= min_support:
            comparison.append((label, support, left, right, right - left))

    return sorted(comparison, key=lambda row: (-abs(row[4]), row[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", choices=MODELS)
    parser.add_argument("second", choices=MODELS)
    parser.add_argument("--min-support", type=int, default=20)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    if args.first == args.second or args.min_support < 1 or args.top < 1:
        parser.error("Choose distinct models and positive support and row limits.")

    rows = compare_classes(DATA, args.first, args.second, args.min_support)
    print(f"{'Class':<26} {'Pages':>5} {args.first:>10} {args.second:>10} {'Gap (pp)':>9}")
    for label, support, left, right, gap in rows[:args.top]:
        print(f"{label:<26} {support:>5} {left:>10.2%} {right:>10.2%} {gap * 100:>+9.2f}")
