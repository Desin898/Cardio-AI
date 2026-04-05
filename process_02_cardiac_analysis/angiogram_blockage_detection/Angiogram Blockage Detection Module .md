# Angiogram Blockage Detection Module

## Objective
This module performs coronary artery analysis using angiogram images.  
It detects potential vessel narrowing (stenosis) and highlights blockage regions to assist clinical interpretation.

This module acts as the **confirmation stage** following ECG-based risk screening.

---

## Overview

The angiogram analysis pipeline is based on a hybrid approach combining:

1. **Deep Learning-based vessel segmentation (DeepSA)**
2. **Geometric analysis for stenosis detection**

This ensures both **accurate vessel extraction** and **interpretable blockage localization**.

---

## Input
- Preprocessed angiogram frames (from Process 01)
- Grayscale angiogram images

---

## Processing Pipeline

### 1. Preprocessing
- Image resizing (512 × 512)
- Morphological enhancement using **Top-Hat filtering**
- Intensity normalization

---

### 2. Vessel Segmentation (DeepSA Model)

- A **U-Net based architecture (DeepSA)** is used for vessel extraction
- The model is initialized with pretrained weights (`.ckpt`)
- Multi-scale fusion strategy is used:
  - Enhanced (Top-Hat) input
  - Standard input
- Outputs are merged using **maximum intensity projection**

Result:
- Clean vessel segmentation mask

---

### 3. Vessel Geometry Analysis (Custom Contribution)

After segmentation, a **custom geometric analysis module** is applied:

- Vessel centerlines are extracted
- Local vessel diameters are computed
- Reference diameter vs narrowed diameter is compared

#### Stenosis Calculation:
- Percentage narrowing is computed using:
  - Minimum diameter
  - Reference vessel width

#### Severity Classification:
- Mild
- Moderate
- Severe

---

### 4. Blockage Localization

- Detected stenosis points are converted into:
  - **(x, y) coordinates**
  - Bounding regions

- Top-K most severe blockages are selected
- Results are visualized using:
  - Colored overlays
  - Percentage labels

---

## Output

The module produces:

1. **Enhanced Subtraction Image**
2. **Vessel Segmentation Mask**
3. **Final Blockage Visualization**

Example Output:
- Highlighted stenosis regions with severity labels (e.g., "78% Severe")

---

## Validation & Evaluation

### Dataset Used
- **ARCADE Dataset** (Annotated coronary angiograms)

### Evaluation Method

- Ground truth bounding boxes were extracted from ARCADE annotations
- Model predictions were compared using:

  - **IoU (Intersection over Union)**
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)

### Metrics Computed

- Precision
- Recall
- F1-Score

---

### Purpose of Evaluation

- Validate correctness of detected blockage locations
- Ensure model predictions align with clinical annotations
- Quantify detection performance objectively

---

## Model Strengths

- Combines **Deep Learning + Analytical Geometry**
- Provides **interpretable outputs** (not just predictions)
- Supports **clinical decision-making**
- Multi-scale segmentation improves robustness

---

## Limitations

- Performance depends on segmentation quality
- Sensitive to low-quality angiogram images
- Requires proper preprocessing from earlier pipeline stages

---

## Integration in Full Pipeline

This module serves as **Stage 2** in the system:

ECG Risk Prediction → Angiogram Analysis → Final Risk Assessment

- ECG identifies high-risk patients
- Angiogram confirms blockage presence and severity

---

## Conclusion

The Angiogram Blockage Detection Module enhances the system by:

- Providing **visual and quantitative evidence** of coronary blockage
- Supporting early diagnosis of coronary artery disease
- Enabling a **multi-modal decision support system**

---

## Notes

- This system is designed as a **decision support tool**
- Final diagnosis should be made by qualified medical professionals