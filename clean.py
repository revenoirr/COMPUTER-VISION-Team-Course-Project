"""Stage 3 — Cleaning: skin / non-skin masks + morphological opening & closing."""

import cv2
import numpy as np

import config


def create_skin_mask(roi: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

    hsv_mask1 = cv2.inRange(hsv, config.SKIN_HSV_LOWER1, config.SKIN_HSV_UPPER1)
    hsv_mask2 = cv2.inRange(hsv, config.SKIN_HSV_LOWER2, config.SKIN_HSV_UPPER2)
    hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)

    ycrcb_mask = cv2.inRange(ycrcb, config.SKIN_YCRCB_LOWER, config.SKIN_YCRCB_UPPER)

    return cv2.bitwise_and(hsv_mask, ycrcb_mask)


def create_non_skin_mask(roi: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, saturation, value = cv2.split(hsv)

    not_skin = cv2.bitwise_not(skin_mask)

    enough_saturation = cv2.inRange(saturation, config.NON_SKIN_MIN_SATURATION, 255)
    enough_brightness = cv2.inRange(value, config.NON_SKIN_MIN_VALUE, 255)

    non_skin_mask = cv2.bitwise_and(not_skin, enough_saturation)
    non_skin_mask = cv2.bitwise_and(non_skin_mask, enough_brightness)

    return non_skin_mask


def _morph_clean(mask: np.ndarray, kernel_small, kernel_big) -> np.ndarray:
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel_small,
        iterations=config.MORPH_OPEN_ITER,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel_big,
        iterations=config.MORPH_CLOSE_ITER,
    )
    return mask


def clean(face_data: list) -> list:
    cleaned = []

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_SMALL)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, config.MORPH_KERNEL_BIG)

    for face in face_data:
        roi = face["roi"]

        skin_mask = create_skin_mask(roi)
        non_skin_mask = create_non_skin_mask(roi, skin_mask)

        skin_mask = _morph_clean(skin_mask, kernel_small, kernel_big)
        non_skin_mask = _morph_clean(non_skin_mask, kernel_small, kernel_big)

        cleaned.append(
            {
                "face_id": face["face_id"],
                "face_rect": face["face_rect"],
                "roi_rect": face["roi_rect"],
                "skin_mask": skin_mask,
                "non_skin_mask": non_skin_mask,
                "roi": roi,
            }
        )

    return cleaned
