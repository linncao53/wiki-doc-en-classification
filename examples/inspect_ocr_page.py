"""Inspect docTR words and bounding boxes for a single upright page image."""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def scale_box(box, source_width, source_height, target_width, target_height):
    """Rescale and clamp an XYXY box to the target coordinate range."""
    x0, y0, x1, y1 = box

    def scale(value, source_size, target_size):
        return max(0, min(target_size, int(value * target_size / source_size)))

    return [
        scale(x0, source_width, target_width),
        scale(y0, source_height, target_height),
        scale(x1, source_width, target_width),
        scale(y1, source_height, target_height),
    ]


def read_words(image):
    import numpy as np
    from doctr.models import ocr_predictor

    predictor = ocr_predictor(pretrained=True, assume_straight_pages=True)
    page = predictor([np.asarray(image)]).pages[0]
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                (x0, y0), (x1, y1) = word.geometry
                yield word.value, scale_box(
                    (x0, y0, x1, y1), 1, 1, image.width, image.height
                )


def save_inspection(image, words, output_dir, source):
    """Write an overlay and the text and word-level boxes derived from it."""
    output_dir.mkdir(parents=True, exist_ok=True)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    records = []
    for word, box in words:
        draw.rectangle(box, outline="#d41455", width=max(2, image.width // 400))
        records.append({
            "word": word,
            "pixel_box": box,
            "layoutxlm_box": scale_box(box, image.width, image.height, 1000, 1000),
        })

    payload = {
        "source": source,
        "width": image.width,
        "height": image.height,
        "distilbert_text": " ".join(record["word"] for record in records),
        "words": records,
    }
    overlay.save(output_dir / "overlay.png")
    (output_dir / "ocr.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="First-page image, e.g. a PNG or JPEG")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    try:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        words = list(read_words(image))
        save_inspection(image, words, args.output_dir, f"docTR on {args.image.name}")
    except ModuleNotFoundError as error:
        parser.exit(1, f"OCR dependency missing: {error}. Install python-doctr.\n")
    except OSError as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {len(words)} words to {args.output_dir}")


if __name__ == "__main__":
    main()
