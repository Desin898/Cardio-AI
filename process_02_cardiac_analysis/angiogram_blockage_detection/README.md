
# Angiogram Blockage Detection Module

## Objective
This module confirms coronary artery blockages using angiogram images.
It is activated only when ECG-based risk is medium or high.

## Input
- Preprocessed angiogram frames (from Process 01)

## Processing
- Vessel segmentation
- Blockage localization
- Stenosis severity estimation

Initial implementation uses traditional image processing.
Advanced implementation integrates DeepSA / U-Net based segmentation.

## Output
Example output:
```json
{
  "patient_id": "P001",
  "blockage_detected": true,
  "stenosis_percentage": 68,
  "affected_artery": "LAD"
}
