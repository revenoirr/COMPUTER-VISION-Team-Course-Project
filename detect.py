"""Stage 4 — Feature detection: skin / non-skin ratios for upper and lower face halves."""

import numpy as np


def ratio(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0

    return np.count_nonzero(mask) / mask.size


def _split_halves(mask: np.ndarray) -> tuple:
    mid = mask.shape[0] // 2
    return mask[:mid, :], mask[mid:, :]


def detect(enhanced: np.ndarray, cleaned_faces: list) -> list:
    detections = []

    for face in cleaned_faces:
        skin_mask = face["skin_mask"]
        non_skin_mask = face["non_skin_mask"]

        upper_skin, lower_skin = _split_halves(skin_mask)
        upper_non_skin, lower_non_skin = _split_halves(non_skin_mask)

        detections.append(
            {
                "face_id": face["face_id"],
                "face_rect": face["face_rect"],
                "skin_ratio": ratio(skin_mask),
                "upper_skin_ratio": ratio(upper_skin),
                "lower_skin_ratio": ratio(lower_skin),
                "non_skin_ratio": ratio(non_skin_mask),
                "upper_non_skin_ratio": ratio(upper_non_skin),
                "lower_non_skin_ratio": ratio(lower_non_skin),
            }
        )

    return detections
