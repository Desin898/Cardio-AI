# ECG Risk Prediction Module

## Objective
This module performs early cardiac risk screening using ECG images.
It estimates the probability of myocardial infarction or abnormal
cardiac patterns.

This module does NOT detect coronary blockages directly.
Instead, it determines whether angiogram analysis is required.

## Input
- Preprocessed ECG images
- ECG image datasets (Normal, MI, Abnormal, History of MI)

## Processing
- ECG image normalization
- Deep learning based classification (CNN)
- Probability-based risk estimation

## Output
Example output:
```json
{
  "patient_id": "P001",
  "ecg_risk_score": 0.82,
  "risk_level": "High"
}
