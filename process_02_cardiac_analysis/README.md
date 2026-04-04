# Process 02 – Cardiac Analysis System

Owner: Desindu

## Overview
This process implements early-stage cardiac risk analysis and coronary
blockage detection following real-world clinical workflow.

The system operates in two stages:
1. ECG-based risk screening
2. Angiogram-based blockage confirmation

This modular design ensures accurate diagnosis while minimizing
unnecessary invasive procedures.

## Submodules
- ECG Risk Prediction
- Angiogram Blockage Detection
- Decision Integration Layer

## Input Sources
- Preprocessed ECG images (from Process 01)
- Preprocessed angiogram frames (from Process 01)

## Outputs
- ECG-based risk probability
- Angiogram-based blockage severity metrics
- Structured output for downstream risk prediction (Process 03)
