"""Illustrate late fusion of aligned text and image scores with synthetic data."""
from math import isclose, isfinite


def fuse_probabilities(text_scores, image_scores, text_weight):
    """Combine probability vectors keyed by document ID and class label."""
    if not 0 <= text_weight <= 1:
        raise ValueError("Text weight must be between zero and one.")
    if not text_scores or text_scores.keys() != image_scores.keys():
        raise ValueError("Text and image scores must cover the same documents.")

    labels = sorted(next(iter(text_scores.values())))
    if not labels:
        raise ValueError("At least one class is required.")

    fused = {}
    for document_id, text in text_scores.items():
        image = image_scores[document_id]
        for scores in (text, image):
            if sorted(scores) != labels:
                raise ValueError(f"Class labels differ for {document_id!r}.")
            if any(not isfinite(p) or not 0 <= p <= 1 for p in scores.values()):
                raise ValueError(f"Invalid probability for {document_id!r}.")
            if not isclose(sum(scores.values()), 1.0, abs_tol=1e-6):
                raise ValueError(f"Probabilities do not sum to one for {document_id!r}.")

        fused[document_id] = {
            label: text_weight * text[label] + (1 - text_weight) * image[label]
            for label in labels
        }
    return fused


if __name__ == "__main__":
    text = {"doc-001": {"Album": 0.65, "Musical": 0.35}}
    image = {"doc-001": {"Musical": 0.80, "Album": 0.20}}
    for document_id, scores in fuse_probabilities(text, image, text_weight=0.5).items():
        label = max(scores, key=scores.get)
        print(f"{document_id}: {label} ({scores[label]:.1%} fused score)")
