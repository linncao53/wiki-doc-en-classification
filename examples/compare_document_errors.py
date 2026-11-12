"""Compare DistilBERT and DiT predictions on the same held-out documents."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def read_predictions(path):
    """Read a CSV keyed by stable document ID."""
    predictions = {}
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = csv.DictReader(source)
        header = rows.fieldnames or []
        id_field = "document_id" if "document_id" in header else "sample_id"
        fields = (id_field, "true_label", "predicted_label")
        if not set(fields).issubset(header) or len(set(header)) != len(header):
            raise ValueError(
                f"{path}: expected unique columns including document_id or sample_id, "
                "true_label, and predicted_label"
            )
        for row in rows:
            document_id, truth, prediction = [(row[field] or "").strip() for field in fields]
            if not all((document_id, truth, prediction)):
                raise ValueError(f"{path}:{rows.line_num}: missing ID or label")
            if document_id in predictions:
                raise ValueError(f"{path}:{rows.line_num}: duplicate ID {document_id!r}")
            predictions[document_id] = (truth, prediction)
    if not predictions:
        raise ValueError(f"{path}: no predictions")
    return predictions


def compare_predictions(distilbert, dit):
    if not distilbert or distilbert.keys() != dit.keys():
        raise ValueError(
            "Expected identical nonempty document sets. "
            f"DistilBERT only: {len(distilbert.keys() - dit.keys())}. "
            f"DiT only: {len(dit.keys() - distilbert.keys())}."
        )

    outcomes = {
        (True, False): "only_distilbert_correct",
        (False, True): "only_dit_correct",
        (True, True): "both_correct",
        (False, False): "both_failed",
    }
    counts = Counter(dict.fromkeys(outcomes.values(), 0))
    for document_id, (truth, text_prediction) in distilbert.items():
        image_truth, image_prediction = dit[document_id]
        if truth != image_truth:
            raise ValueError(f"Ground-truth labels differ for {document_id!r}")
        outcome = outcomes[text_prediction == truth, image_prediction == truth]
        counts[outcome] += 1

    return {"documents": len(distilbert), **counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distilbert_csv", type=Path)
    parser.add_argument("dit_csv", type=Path)
    args = parser.parse_args()
    try:
        counts = compare_predictions(
            read_predictions(args.distilbert_csv), read_predictions(args.dit_csv)
        )
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
