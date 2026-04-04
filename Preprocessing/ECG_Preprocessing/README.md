# ECG Image Preprocessing Pipeline

## Overview
This directory contains the **ECG image preprocessing pipeline** developed as part of  
**Process 01 – ECG and Angiogram Preprocessing System** for the  
*AI-Driven Coronary Disease Detection and Decision Support System*.

The preprocessing pipeline transforms raw ECG images into **standardized, machine-learning-ready signals** that are later used by the **Risk Prediction module**.

All preprocessing steps are implemented and demonstrated in the accompanying Jupyter Notebook.

---

## Objectives
- Remove noise and grid artifacts from scanned ECG images
- Extract clinically meaningful ECG signals from 12-lead layouts
- Standardize signal length and scale for downstream ML models
- Export structured numerical data in CSV format

---

## Preprocessing Steps Summary

| Step | Technique Used | Why It Was Chosen |
|-----|----------------|------------------|
| Color Removal | Grayscale conversion | ECG images do not contain useful color information |
| Noise Removal | Median filtering | Effective at removing grid lines and scanning noise |
| Segmentation | Adaptive thresholding | Handles non-uniform illumination across ECG scans |
| Cleanup | Morphological operations | Removes residual artifacts after thresholding |
| Lead Extraction | 12-lead grid-based cropping | Matches standard clinical ECG layout (4×3 leads) |
| Feature Creation | Signal centerline extraction | Converts waveform image to numerical signal |
| Standardization | Resampling (linear interpolation) | Ensures fixed-length signals for ML input |
| Output Generation | CSV per class | Enables easy downstream model integration |

---

## Technical Highlights
- **Adaptive Thresholding**: Robust against lighting variations in scanned ECG reports
- **Grid-Based Cropping**: Assumes standard ECG paper layout (12 leads)
- **Signal Normalization**: All extracted signals scaled to the \[0,1\] range
- **Fixed-Length Resampling**: Guarantees consistent input dimensionality

---

## Output Format
The preprocessing pipeline generates CSV files with the following structure:
filename, Lead1_0, Lead1_1, ..., Lead12_N
