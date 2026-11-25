"""Compare confidence and random routing on synthetic aligned logits."""

import math
import random


def log_softmax(logits, temperature=1.0):
    shifted = [(score - max(logits)) / temperature for score in logits]
    normalizer = math.log(sum(math.exp(score) for score in shifted))
    return [score - normalizer for score in shifted]


def page_predictions(text_logits, image_logits, temperature, image_weight):
    text = log_softmax(text_logits, temperature)
    image = log_softmax(image_logits)
    classes = range(len(text))
    text_prediction = max(classes, key=text.__getitem__)
    blend_prediction = max(
        classes, key=lambda i: (1 - image_weight) * text[i] + image_weight * image[i]
    )
    return math.exp(text[text_prediction]), text_prediction, blend_prediction


def evaluate(pages, labels, routed, class_count):
    predictions = [
        blend if index in routed else text
        for index, (_, text, blend) in enumerate(pages)
    ]
    f1 = []
    for label in range(class_count):
        actual = labels.count(label)
        predicted = predictions.count(label)
        correct = sum(y == p == label for y, p in zip(labels, predictions))
        f1.append(2 * correct / (actual + predicted) if actual + predicted else 0)
    return sum(f1) / class_count


def main():
    # Each row contains the gold label, then the two models' logits in class order.
    examples = [
        (0, [8, 0, -1], [0, 4, 0]),
        (1, [2, 1, 0], [0, 5, 0]),
        (2, [1, 0, 2], [0, 0, 5]),
        (0, [2, 1, 0], [0, 4, 0]),
        (1, [0, 8, 0], [4, 0, 0]),
        (2, [0, 2, 1], [0, 0, 5]),
    ]
    labels = [label for label, _, _ in examples]
    pages = [
        page_predictions(text, image, temperature=1.30, image_weight=0.40)
        for _, text, image in examples
    ]
    routed = {i for i, (confidence, _, _) in enumerate(pages) if confidence < 0.98}
    all_pages = range(len(pages))
    rng = random.Random(42)
    random_scores = [
        evaluate(pages, labels, set(rng.sample(all_pages, len(routed))), 3)
        for _ in range(100)
    ]
    print(f"Pages selected for DiT: {len(routed)}/{len(pages)}")
    print(f"DistilBERT-only macro-F1: {evaluate(pages, labels, set(), 3):.4f}")
    print(f"Confidence routing macro-F1: {evaluate(pages, labels, routed, 3):.4f}")
    print(f"Matched random mean macro-F1: {sum(random_scores) / len(random_scores):.4f}")
    print(f"Always-blend macro-F1: {evaluate(pages, labels, set(all_pages), 3):.4f}")


if __name__ == "__main__":
    main()
