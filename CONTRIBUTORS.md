# Contribution Statement

**Project:** Face Region Analysis — Face Mask Detection
**Theme:** 15 — Human Systems / Face Region Analysis
**Course:** Computer Vision (Team Course Project)

This document declares the role assignments and individual contributions of every team member, in accordance with the project brief. It is intended to be included in the appendix of the final PDF report.

---

## Team & Roles

| Name | Role | Pipeline Ownership | GitHub |
|------|------|--------------------|--------|
| Alexandra Nerush | Lead CV Engineer | Detect + Decision | [@alexanerush](https://github.com/alexanerush) |
| Sofya Asinskaya | Image Processing Specialist | Enhance + Segment | [@ossofi](https://github.com/ossofi) |
| Fiodar Viachorka | Morphology & Report Lead | Clean + Visualization | [@revenoirr](https://github.com/revenoirr) |

---

## Individual Contributions

### Alexandra Nerush — Lead CV Engineer

Responsible for the final two stages of the pipeline and overall end-to-end integration.

- Implemented **Stage 4 — Feature Detection** ([detect.py](detect.py))
  - Computation of skin and non-skin pixel ratios
  - Upper / lower face half splitting and per-half ratio extraction
- Implemented **Stage 5 — Decision** ([decide.py](decide.py))
  - Rule-based classifier with five branches covering MASK, PARTIAL MASK, and NO MASK
  - Color coding of bounding boxes per decision class
- Tuned classification thresholds in [config.py](config.py) based on observed metric distributions across test images
- Integrated all five pipeline stages in [main.py](main.py) and ensured the system runs end-to-end on real images
- Validated the system on multi-person scenes and verified that per-face decisions are independent

### Sofya Asinskaya — Image Processing Specialist

Responsible for image preprocessing and face localization.

- Implemented **Stage 1 — Enhancement** ([enhance.py](enhance.py))
  - CLAHE contrast enhancement applied to the LAB luminance channel
  - Fast non-local-means colored denoising
  - Gamma correction for brightness normalization
- Implemented **Stage 2 — Segmentation** ([segment.py](segment.py))
  - Primary DNN SSD face detector (`res10_300x300_ssd_iter_140000`)
  - Haar Cascade fallback chain (alt2, default, alt, profile) when DNN is unavailable or yields no detections
  - Face ROI extraction using fractional crops of the detected bounding box to isolate the nose and mouth region
- Configured detection parameters (DNN confidence threshold, Haar scale factor, minimum face size) in [config.py](config.py)
- Tested face detection on single-face and multi-face images across varying lighting conditions

### Fiodar Viachorka — Morphology & Report Lead

Responsible for noise cleaning, visualization, documentation, and repository setup.

- Implemented **Stage 3 — Cleaning** ([clean.py](clean.py))
  - Dual color-space skin masking combining HSV (two ranges to handle hue wrap-around) and YCrCb
  - Complementary non-skin mask construction using saturation and value gates
  - Morphological opening to remove isolated noise pixels
  - Morphological closing to fill gaps and smooth segmented regions
- Implemented **Visualization** ([visualize.py](visualize.py))
  - Per-stage image output (original, enhanced, segmentation, cleaned, detection)
  - Labeled detection rendering with color-coded bounding boxes and per-face metric overlays
  - FACE NOT DETECTED fallback annotation
- Authored the full team report ([report.md](report.md))
  - Abstract, problem description, methods, stage-by-stage results, limitations, conclusion
  - Embedded result screenshots covering all four decision categories
- Authored the project [README.md](README.md) with pipeline overview, installation, and usage instructions
- Set up the repository: directory layout, [requirements.txt](requirements.txt), [.gitignore](.gitignore), `report_images/`
- Prepared and curated result screenshots used in the report and README

---

## Shared Responsibilities

All three team members participated in the following collaborative activities:

- Reviewed each other's code and pipeline stages
- Tested the integrated system on shared image samples
- Discussed and agreed on threshold values and decision rules
- Prepared and rehearsed the team presentation

---

## Signatures

By signing below, each team member confirms that the contributions listed above accurately reflect the work they performed on this project.

| Name | Role | Signature | Date |
|------|------|-----------|------|
| Alexandra Nerush | Lead CV Engineer | ______________________ | __________ |
| Sofya Asinskaya | Image Processing Specialist | ______________________ | __________ |
| Fiodar Viachorka | Morphology & Report Lead | ______________________ | __________ |
