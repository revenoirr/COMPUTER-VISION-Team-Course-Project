"""Stage 2 — Face segmentation: DNN SSD detector with Haar Cascade fallback."""

import os

import cv2
import numpy as np

import config


def _filter_by_relative_size(faces: list, image_shape: tuple) -> list:
    img_h, img_w = image_shape[:2]
    min_dim = min(img_h, img_w) * config.MIN_FACE_FRAC
    return [(x, y, w, h) for (x, y, w, h) in faces if w >= min_dim and h >= min_dim]


def _detect_with_dnn(enhanced: np.ndarray) -> list:
    if not (os.path.exists(config.DNN_PROTO_PATH) and os.path.exists(config.DNN_MODEL_PATH)):
        return []

    img_h, img_w = enhanced.shape[:2]
    face_rects = []

    try:
        net = cv2.dnn.readNetFromCaffe(config.DNN_PROTO_PATH, config.DNN_MODEL_PATH)

        blob = cv2.dnn.blobFromImage(
            cv2.resize(enhanced, config.DNN_INPUT_SIZE),
            1.0,
            config.DNN_INPUT_SIZE,
            config.DNN_MEAN,
        )

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence <= config.DNN_CONFIDENCE_THRESHOLD:
                continue

            box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
            x1, y1, x2, y2 = box.astype("int")

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w - 1, x2)
            y2 = min(img_h - 1, y2)

            w = x2 - x1
            h = y2 - y1

            if w > config.DNN_MIN_FACE_SIZE and h > config.DNN_MIN_FACE_SIZE:
                face_rects.append((x1, y1, w, h))

        if face_rects:
            print("  Face detector used: DNN SSD")

    except Exception as e:
        print(f"  [!] DNN detector failed: {e}")

    return face_rects


def _detect_with_haar(enhanced: np.ndarray) -> list:
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    for cascade_name in config.HAAR_CASCADES:
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascade_name)

        detected = cascade.detectMultiScale(
            gray,
            scaleFactor=config.HAAR_SCALE_FACTOR,
            minNeighbors=config.HAAR_MIN_NEIGHBORS,
            minSize=config.HAAR_MIN_SIZE,
        )

        valid = _filter_by_relative_size(
            [tuple(rect) for rect in detected],
            enhanced.shape,
        )

        if valid:
            print(f"  Face detector used: {cascade_name}")
            return valid

    return []


def _extract_roi(enhanced: np.ndarray, face_rect: tuple) -> tuple:
    img_h, img_w = enhanced.shape[:2]
    x, y, w, h = face_rect

    x = max(0, int(x))
    y = max(0, int(y))
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)

    roi_x1 = max(0, x + int(w * config.ROI_LEFT_FRAC))
    roi_x2 = min(img_w, x + int(w * config.ROI_RIGHT_FRAC))
    roi_y1 = max(0, y + int(h * config.ROI_TOP_FRAC))
    roi_y2 = min(img_h, y + int(h * config.ROI_BOTTOM_FRAC))

    roi = enhanced[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_rect = (roi_x1, roi_y1, roi_x2 - roi_x1, roi_y2 - roi_y1)

    return (x, y, w, h), roi_rect, roi


def segment(enhanced: np.ndarray) -> list:
    face_rects = _detect_with_dnn(enhanced)
    face_rects = _filter_by_relative_size(face_rects, enhanced.shape)

    if not face_rects:
        face_rects = _detect_with_haar(enhanced)

    face_data = []

    for face_id, rect in enumerate(face_rects, start=1):
        face_rect, roi_rect, roi = _extract_roi(enhanced, rect)

        if roi.size == 0:
            continue

        face_data.append(
            {
                "face_id": face_id,
                "face_rect": face_rect,
                "roi_rect": roi_rect,
                "roi": roi,
            }
        )

    return face_data
