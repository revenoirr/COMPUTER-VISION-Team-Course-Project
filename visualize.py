"""Visualization — write per-stage images and the final labeled detection."""

import os

import cv2
import numpy as np

import config


def _save_stage_image(output_dir: str, base: str, suffix: str, image: np.ndarray) -> None:
    cv2.imwrite(os.path.join(output_dir, f"{base}_{suffix}.jpg"), image)


def _build_segmentation_image(enhanced: np.ndarray, cleaned_faces: list) -> np.ndarray:
    seg_vis = np.zeros(enhanced.shape[:2], dtype=np.uint8)

    for face in cleaned_faces:
        roi_x, roi_y, roi_w, roi_h = face["roi_rect"]
        combined = cv2.bitwise_or(face["skin_mask"], face["non_skin_mask"])

        target_h = min(roi_h, seg_vis.shape[0] - roi_y)
        target_w = min(roi_w, seg_vis.shape[1] - roi_x)

        if target_h > 0 and target_w > 0:
            seg_vis[
                roi_y:roi_y + target_h,
                roi_x:roi_x + target_w,
            ] = combined[:target_h, :target_w]

    return seg_vis


def _draw_face_label(image: np.ndarray, result: dict) -> None:
    face_id = result["face_id"]
    x, y, w, h = result["face_rect"]
    color = result["color"]
    label = result["label"]

    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    text_y1 = max(0, y - 34)
    cv2.rectangle(
        image,
        (x, text_y1),
        (x + max(w, 180), y),
        color,
        -1,
    )

    cv2.putText(
        image,
        f"Person {face_id}: {label}",
        (x + 4, max(18, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
    )

    metrics = [
        ("skin", result["skin_ratio"]),
        ("upper", result["upper_skin_ratio"]),
        ("lower", result["lower_skin_ratio"]),
    ]
    for i, (name, value) in enumerate(metrics):
        offset = 38 - i * 15
        cv2.putText(
            image,
            f"{name}={value:.2f}",
            (x + 4, y + h - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )


def _draw_not_detected(image: np.ndarray) -> None:
    cv2.putText(
        image,
        "FACE NOT DETECTED",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        config.COLOR_NOT_DETECTED,
        2,
    )


def visualize(original, enhanced, cleaned_faces, results, output_dir, img_name):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(img_name)[0]

    _save_stage_image(output_dir, base, "1_original", original)
    _save_stage_image(output_dir, base, "2_enhanced", enhanced)

    seg_vis = _build_segmentation_image(enhanced, cleaned_faces)
    _save_stage_image(output_dir, base, "3_segmentation", seg_vis)
    _save_stage_image(output_dir, base, "4_cleaned", seg_vis)

    result_img = original.copy()

    if not results:
        _draw_not_detected(result_img)
    else:
        for result in results:
            _draw_face_label(result_img, result)

    _save_stage_image(output_dir, base, "5_detection", result_img)

    return result_img
