"""Configuration constants for the face mask detection pipeline."""

import numpy as np

# ---------- Stage 1: Enhance ----------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

DENOISE_H = 7
DENOISE_H_COLOR = 7
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21

GAMMA = 1.15


# ---------- Stage 2: Segment ----------
DNN_PROTO_PATH = "deploy.prototxt"
DNN_MODEL_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
DNN_INPUT_SIZE = (300, 300)
DNN_MEAN = (104.0, 177.0, 123.0)
DNN_CONFIDENCE_THRESHOLD = 0.35
DNN_MIN_FACE_SIZE = 20

HAAR_CASCADES = [
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_profileface.xml",
]
HAAR_SCALE_FACTOR = 1.04
HAAR_MIN_NEIGHBORS = 3
HAAR_MIN_SIZE = (25, 25)

# Face ROI as fraction of detected face bounding box
ROI_LEFT_FRAC = 0.10
ROI_RIGHT_FRAC = 0.90
ROI_TOP_FRAC = 0.35
ROI_BOTTOM_FRAC = 0.92


# ---------- Stage 3: Clean ----------
SKIN_HSV_LOWER1 = np.array([0, 25, 50], dtype=np.uint8)
SKIN_HSV_UPPER1 = np.array([25, 255, 255], dtype=np.uint8)
SKIN_HSV_LOWER2 = np.array([160, 25, 50], dtype=np.uint8)
SKIN_HSV_UPPER2 = np.array([180, 255, 255], dtype=np.uint8)

SKIN_YCRCB_LOWER = np.array([0, 133, 77], dtype=np.uint8)
SKIN_YCRCB_UPPER = np.array([255, 173, 127], dtype=np.uint8)

NON_SKIN_MIN_SATURATION = 20
NON_SKIN_MIN_VALUE = 35

MORPH_KERNEL_SMALL = (3, 3)
MORPH_KERNEL_BIG = (5, 5)
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 2


# ---------- Stage 5: Decide thresholds ----------
MASK_SKIN_RATIO_MAX = 0.22
MASK_LOWER_SKIN_RATIO_MAX = 0.18

PARTIAL_LOWER_NON_SKIN_MIN = 0.35
PARTIAL_SKIN_DROP_MIN = 0.25
PARTIAL_UPPER_SKIN_MIN_HIGH = 0.45
PARTIAL_LOWER_SKIN_MAX_HIGH = 0.35
PARTIAL_LOWER_SKIN_MAX_LOW = 0.20
PARTIAL_UPPER_SKIN_MIN_LOW = 0.35

NO_MASK_LOWER_SKIN_MIN = 0.45
NO_MASK_SKIN_RATIO_MIN_HIGH = 0.45
NO_MASK_SKIN_RATIO_MIN_SOLO = 0.60


# ---------- Output colors (BGR) ----------
COLOR_MASK = (0, 200, 0)
COLOR_PARTIAL = (0, 220, 255)
COLOR_NO_MASK = (0, 0, 220)
COLOR_NOT_DETECTED = (0, 0, 255)


# ---------- Output ----------
DEFAULT_OUTPUT_DIR = "output"
