"""Stage 5 — Decision: rule-based classifier into MASK / PARTIAL MASK / NO MASK."""

import config


def _classify(skin_ratio, upper_skin_ratio, lower_skin_ratio,
              lower_non_skin_ratio) -> tuple:
    skin_drop = upper_skin_ratio - lower_skin_ratio
    mask_on_lower = lower_non_skin_ratio > config.PARTIAL_LOWER_NON_SKIN_MIN
    very_low_lower_skin = lower_skin_ratio < config.PARTIAL_LOWER_SKIN_MAX_LOW

    if (skin_ratio < config.MASK_SKIN_RATIO_MAX
            and lower_skin_ratio < config.MASK_LOWER_SKIN_RATIO_MAX):
        return "MASK", config.COLOR_MASK

    if (lower_skin_ratio < config.MASK_LOWER_NEAR_ZERO
            and skin_drop > config.MASK_SKIN_DROP_STRONG):
        return "MASK", config.COLOR_MASK

    if mask_on_lower and skin_drop > config.PARTIAL_SKIN_DROP_MIN:
        return "PARTIAL MASK", config.COLOR_PARTIAL

    if (upper_skin_ratio > config.PARTIAL_UPPER_SKIN_MIN_HIGH
            and lower_skin_ratio < config.PARTIAL_LOWER_SKIN_MAX_HIGH):
        return "PARTIAL MASK", config.COLOR_PARTIAL

    if (very_low_lower_skin
            and upper_skin_ratio > config.PARTIAL_UPPER_SKIN_MIN_LOW):
        return "PARTIAL MASK", config.COLOR_PARTIAL

    if (lower_skin_ratio > config.NO_MASK_LOWER_SKIN_MIN
            and skin_ratio > config.NO_MASK_SKIN_RATIO_MIN_HIGH):
        return "NO MASK", config.COLOR_NO_MASK

    if skin_ratio > config.NO_MASK_SKIN_RATIO_MIN_SOLO:
        return "NO MASK", config.COLOR_NO_MASK

    return "PARTIAL MASK", config.COLOR_PARTIAL


def decide(detections: list) -> list:
    results = []

    for detection in detections:
        label, color = _classify(
            detection["skin_ratio"],
            detection["upper_skin_ratio"],
            detection["lower_skin_ratio"],
            detection["lower_non_skin_ratio"],
        )

        results.append(
            {
                **detection,
                "label": label,
                "color": color,
            }
        )

    return results
