# FACE REGION ANALYSIS: FACE MASK DETECTION SYSTEM

## Computer Vision Final Project

### Theme 15 – Face Region Analysis

---

# Abstract

This project presents a computer vision system designed to analyze facial regions and determine whether a person is wearing a face mask correctly, partially wearing a face mask, or not wearing a face mask at all. The implemented solution follows the required computer vision pipeline:

**Image → Enhance → Segment → Clean → Detect → Decision**

The system processes input images, detects one or multiple faces, extracts facial features, analyzes mask coverage, and generates a final classification result. The developed application supports multiple people within a single image and provides visual outputs for every stage of the processing pipeline.

---

# 1. Introduction

Face mask detection is an important computer vision problem that became highly relevant during global public health events. Automatic mask detection systems can be used in airports, hospitals, educational institutions, transportation systems, and public facilities.

The objective of this project is to build a complete computer vision application capable of analyzing facial regions and determining mask usage status based on visual information.

Unlike simple image classification approaches, this project implements all stages of a classical computer vision pipeline and demonstrates image enhancement, segmentation, cleaning, feature extraction, and decision-making techniques.

---

# 2. Project Objectives

The main objectives of the project are:

* Detect human faces in digital images
* Support multiple faces in a single image
* Enhance image quality before analysis
* Segment facial regions of interest
* Remove segmentation noise
* Extract visual mask-related features
* Classify detected faces into predefined categories
* Produce interpretable visual results

The system must classify every detected face into one of four categories:

* MASK
* PARTIAL MASK
* NO MASK
* FACE NOT DETECTED

---

# 3. System Requirements

The project requirements specify the following computer vision workflow:

**Image → Enhance → Segment → Clean → Detect → Decision**

Every stage must be implemented and clearly demonstrated.

Input:

* JPG images
* PNG images
* Multiple image processing support

Output:

* Enhanced image
* Segmentation result
* Cleaned mask
* Detection visualization
* Final classification decision

---

# 4. Technologies Used

The implementation was developed using Python and OpenCV.

### Programming Language

Python 3

### Libraries

OpenCV

Used for:

* Image processing
* Face detection
* Morphological operations
* Color space transformations

NumPy

Used for:

* Matrix operations
* Numerical computations
* Pixel statistics

### Face Detection Models

#### Primary Detector

OpenCV DNN Face Detector

Model:

* deploy.prototxt
* res10_300x300_ssd_iter_140000.caffemodel

#### Secondary Detector

OpenCV Haar Cascade Classifiers

Used as a fallback mechanism when DNN detection fails.

---

# 5. System Architecture

The developed system consists of five sequential stages.

## Stage 1 – Image Enhancement

The enhancement stage improves image quality before analysis.

### CLAHE Contrast Enhancement

The image is converted to LAB color space.

The luminance channel is enhanced using:

Contrast Limited Adaptive Histogram Equalization (CLAHE)

Benefits:

* Improves local contrast
* Preserves image details
* Improves visibility under poor lighting

### Noise Reduction

Fast Non-Local Means Denoising is applied.

Benefits:

* Removes sensor noise
* Preserves image edges
* Improves segmentation quality

### Gamma Correction

Gamma correction is used to normalize brightness.

Benefits:

* Better shadow visibility
* Improved feature extraction

Output:

Enhanced image ready for analysis.

---

## Stage 2 – Face Segmentation

The segmentation stage identifies face regions.

### Deep Neural Network Face Detector

The system first attempts face detection using a DNN SSD detector.

Advantages:

* Higher accuracy
* Better robustness
* Better support for different face sizes

Detection confidence threshold:

0.35

### Haar Cascade Fallback

If DNN detection fails, the system automatically switches to Haar Cascade classifiers.

Available cascades:

* Frontal Face Default
* Frontal Face Alt
* Frontal Face Alt2
* Profile Face

This increases robustness and reliability.

### Multi-Face Support

The detector processes all detected faces independently.

Each face receives:

* Face ID
* Bounding Box
* Region of Interest (ROI)

Example:

Person 1

Person 2

Person 3

Each face is analyzed separately.

---

## Stage 3 – Cleaning

The cleaning stage removes noise from segmented regions.

### Skin Detection

Skin regions are detected using two color spaces.

#### HSV Color Space

Used because skin tones occupy relatively stable HSV ranges.

#### YCrCb Color Space

Provides additional robustness against illumination changes.

The final skin mask is obtained by combining both methods.

### Non-Skin Detection

The system also creates a complementary non-skin mask.

This allows identification of:

* Face masks
* Dark coverings
* Colored coverings

### Morphological Processing

Two operations are applied.

#### Opening

Removes isolated noise pixels.

#### Closing

Fills small gaps and smooths segmented regions.

Benefits:

* Cleaner masks
* More stable measurements
* Reduced false positives

---

## Stage 4 – Feature Detection

The detection stage extracts numerical measurements.

For every face the system calculates:

### Skin Ratio

Percentage of skin pixels inside the facial ROI.

### Upper Skin Ratio

Percentage of visible skin in the upper facial region.

### Lower Skin Ratio

Percentage of visible skin in the lower facial region.

### Non-Skin Ratio

Percentage of non-skin pixels.

### Upper Non-Skin Ratio

Non-skin percentage in the upper face.

### Lower Non-Skin Ratio

Non-skin percentage in the lower face.

These values are used during classification.

---

## Stage 5 – Decision Making

The final classification is performed using a rule-based classifier.

### MASK

The face is classified as MASK when:

* Lower face skin visibility is extremely low
* Nose and mouth areas appear covered
* Lower region contains mostly non-skin pixels

Output color:

Green

---

### NO MASK

The face is classified as NO MASK when:

* Significant skin is visible across the face
* Both upper and lower facial regions remain visible

Output color:

Red

---

### PARTIAL MASK

The face is classified as PARTIAL MASK when:

* Mask coverage is incomplete
* The lower face is only partially covered
* Upper and lower visibility differ significantly

Output color:

Yellow

---

### FACE NOT DETECTED

If no face can be detected:

Output:

FACE NOT DETECTED

Possible reasons:

* Low image quality
* Extreme rotations
* Occlusions
* Poor lighting

---

# 6. Output Generation

For every image the system automatically generates:

### Original Image

Input image before processing.

### Enhanced Image

Result after enhancement stage.

### Segmentation Result

Detected skin and non-skin regions.

### Cleaned Mask

Noise-reduced segmentation output.

### Detection Result

Final image containing:

* Bounding boxes
* Person identifiers
* Classification labels
* Statistical measurements

---

# 7. Experimental Results

The system was tested using three categories of images.

## Category 1 – No Mask

Expected Result:

NO MASK

Observed Result:

The majority of images were correctly classified. The face is fully visible, the skin ratio is high across both upper and lower halves, and the classifier confidently assigns the NO MASK label (red bounding box).

![No mask detection example](report_images/no_mask.jpg)

Measured values: skin = 0.93, upper = 1.00, lower = 0.87 — clearly above the NO MASK thresholds.

---

## Category 2 – Full Mask

Expected Result:

MASK

Observed Result:

The system successfully detected covered mouth and nose regions. With the mask in place, the lower half of the face contains almost no skin pixels while the upper half still shows skin, producing a confident MASK classification (green bounding box).

![Full mask detection example](report_images/mask.jpg)

Measured values: skin = 0.13, upper = 0.27, lower = 0.00 — well below the MASK thresholds.

---

## Category 3 – Partial Mask

Expected Result:

PARTIAL MASK

Observed Result:

The classifier identified incomplete mask coverage and distinguished it from both full mask and no mask conditions. In the example below the mask is pulled down below the nose: the upper half of the face shows high skin visibility while the lower half is partially covered, triggering the PARTIAL MASK rule (yellow bounding box).

![Partial mask detection example](report_images/partial_mask.jpg)

Measured values: skin = 0.55, upper = 0.95, lower = 0.16 — a large upper-vs-lower skin drop that matches the partial-coverage rule.

---

## Multi-Person Testing

The system was tested on images containing multiple individuals.

Results:

* Face detection successfully identified multiple faces
* Independent classification was performed for each person
* Unique labels were assigned to every detected face

Example:

![Multi-person detection example](report_images/multi_person.jpg)

In this scene the pipeline detected two faces and classified them independently:

* Person 1: NO MASK (skin = 0.68, upper = 0.72, lower = 0.64)
* Person 2: PARTIAL MASK (skin = 0.66, upper = 1.00, lower = 0.34)

This demonstrates that the system applies the classification rules per-face, so a single image can yield different labels for different individuals.

---

# 8. Advantages of the Proposed Solution

The developed system provides several advantages.

### Complete Pipeline

All required stages were implemented.

### Multi-Face Support

Multiple individuals can be analyzed simultaneously.

### Hybrid Face Detection

Combines DNN and Haar Cascade approaches.

### Robust Segmentation

Uses two color spaces for improved skin detection.

### Explainable Classification

Every decision is based on measurable visual features.

### Visual Outputs

Every processing stage can be inspected and verified.

### Lightweight Architecture

Runs efficiently on standard hardware.

---

# 9. Limitations

Several limitations remain.

### Extreme Head Rotations

Profile faces may reduce detection accuracy.

### Low Resolution Images

Small faces contain fewer useful features.

### Complex Lighting

Strong shadows may affect skin segmentation.

### Skin-Colored Masks

Masks with colors similar to human skin may reduce classification accuracy.

### Heavy Occlusions

Scarves, hands, or other objects can interfere with analysis.

---

# 10. Future Improvements

Potential future improvements include:

* Deep learning mask classification models
* Real-time webcam processing
* Face tracking in video streams
* Confidence scoring
* CNN-based segmentation
* Dataset-based training and evaluation
* Performance benchmarking

---

# 11. Conclusion

A complete computer vision system for face mask detection was successfully developed.

The solution follows the required workflow:

**Image → Enhance → Segment → Clean → Detect → Decision**

The system performs image enhancement, face detection, segmentation, noise removal, feature extraction, and final classification.

The application supports both single-face and multi-face images and classifies each detected individual as:

* MASK
* PARTIAL MASK
* NO MASK
* FACE NOT DETECTED

Experimental testing demonstrated that the developed solution satisfies the project requirements and successfully applies classical computer vision techniques to a real-world image analysis problem.
