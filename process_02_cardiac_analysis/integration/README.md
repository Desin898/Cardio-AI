
# Decision Integration Layer

## Purpose
This layer integrates ECG risk prediction and angiogram blockage detection
to produce structured outputs for system-wide risk prediction.

## Logic
- Low ECG risk → No angiogram
- Medium / High ECG risk → Angiogram analysis
- Blockage severity contributes to final risk prediction

## Output
Unified schema passed to Process 03 (Risk Prediction Model)
