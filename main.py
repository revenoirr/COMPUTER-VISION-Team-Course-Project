"""
Face Region Analysis — Computer Vision Team Project (Theme 15)
Pipeline: image → enhance → segment → clean → detect → decision

Classes:
- MASK          -> green indicator
- NO MASK       -> red indicator
- PARTIAL MASK  -> yellow indicator
- FACE NOT DETECTED

Usage:
    python main.py image1.jpg image2.jpg ...
"""

import os
import sys

import cv2

import config
from enhance import enhance
from segment import segment
from clean import clean
from detect import detect
from decide import decide
from visualize import visualize


def run_pipeline(image_path: str, output_dir: str = config.DEFAULT_OUTPUT_DIR) -> list:
    image = cv2.imread(image_path)

    if image is None:
        print(f"[ERROR] Cannot load: {image_path}")
        return []

    img_name = os.path.basename(image_path)

    print(f"\n{'=' * 50}")
    print(f"Processing: {img_name}")

    enhanced = enhance(image)
    print("  [1/5] Enhance ✓")

    face_data = segment(enhanced)
    print(f"  [2/5] Segment ✓  ({len(face_data)} faces)")

    if len(face_data) == 0:
        print("  [!] Face not detected. Image may be unclear, rotated, or low quality.")
        visualize(image, enhanced, [], [], output_dir, img_name)
        return ["FACE NOT DETECTED"]

    cleaned = clean(face_data)
    print("  [3/5] Clean ✓")

    detections = detect(enhanced, cleaned)
    print("  [4/5] Detect ✓")

    results = decide(detections)
    print("  [5/5] Decide ✓")

    visualize(image, enhanced, cleaned, results, output_dir, img_name)

    labels = []

    for result in results:
        face_id = result["face_id"]
        label = result["label"]

        labels.append(f"Person {face_id}: {label}")

        print(
            f"  → Person {face_id}: {label} "
            f"(skin={result['skin_ratio']:.2f}, "
            f"upper={result['upper_skin_ratio']:.2f}, "
            f"lower={result['lower_skin_ratio']:.2f}, "
            f"non_skin={result['non_skin_ratio']:.2f})"
        )

    return labels


def main(argv: list) -> int:
    images = argv[1:]

    if not images:
        print("Usage: python main.py image1.jpg image2.jpg ...")
        print("Example: python main.py photo.jpg")
        return 0

    all_results = {}

    for img_path in images:
        all_results[img_path] = run_pipeline(img_path)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for path, labels in all_results.items():
        print(f"  {os.path.basename(path)}: {', '.join(labels)}")

    print(f"\nOutput images saved to: ./{config.DEFAULT_OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
