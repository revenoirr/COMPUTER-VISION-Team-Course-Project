"""Stage 1 — Image enhancement: CLAHE + denoising + gamma correction."""

import cv2
import numpy as np

import config


def enhance(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    l_clahe = clahe.apply(l)

    lab_enhanced = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    enhanced = cv2.fastNlMeansDenoisingColored(
        enhanced,
        None,
        config.DENOISE_H,
        config.DENOISE_H_COLOR,
        config.DENOISE_TEMPLATE_WINDOW,
        config.DENOISE_SEARCH_WINDOW,
    )

    inv_gamma = 1.0 / config.GAMMA
    table = np.array(
        [(i / 255.0) ** inv_gamma * 255 for i in range(256)]
    ).astype("uint8")

    enhanced = cv2.LUT(enhanced, table)

    return enhanced
