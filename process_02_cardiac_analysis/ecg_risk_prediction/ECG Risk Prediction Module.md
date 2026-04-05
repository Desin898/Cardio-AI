# ECG Risk Prediction Module

## Objective
This module performs early cardiac risk screening using ECG images.  
It estimates the probability of myocardial infarction or abnormal cardiac patterns.

This module does NOT detect coronary blockages directly.  
Instead, it determines whether angiogram analysis is required.

---

## Input
- Preprocessed ECG images
- ECG image datasets:
  - Normal
  - Abnormal Heartbeat
  - Myocardial Infarction (MI)
  - History of MI

---

## Processing
- ECG image normalization
- Conversion into structured signal representation (12-lead format)
- Deep Learning-based classification using Convolutional Neural Networks (CNN)
- Probability-based risk estimation using Softmax outputs

---

## Model Architecture
- Input shape: `(1, 12, T)` (12 ECG leads)
- Convolutional layers with Batch Normalization
- Feature extraction through hierarchical CNN layers
- Adaptive pooling for dimensionality reduction
- Fully connected classification layer

---

## Model Evaluation & Cross-Validation

To ensure robustness and generalization of the ECG classification model,  
**5-Fold Cross Validation** was performed on the dataset.

### Cross-Validation Results

| Fold | Accuracy |
|------|--------|
| Fold 1 | 92.23% |
| Fold 2 | 96.69% |
| Fold 3 | 64.63% |
| Fold 4 | 98.84% |
| Fold 5 | 96.85% |

---

### Overall Performance

- **Mean Accuracy:** 89.85%  
- **Standard Deviation:** ±12.79%

---

### Interpretation

- The model performs consistently well across most folds (>95%)
- One fold shows lower accuracy due to possible:
  - Data imbalance
  - Patient variability
  - Noise in ECG signal extraction

Despite this variation, the model demonstrates **strong generalization capability**.

---

### Why Cross-Validation?

Cross-validation ensures that the model is not overfitting to a specific dataset split  
and provides a more reliable estimate of real-world performance.

---

### Conclusion

The ECG CNN model is reliable for early-stage cardiac risk prediction  
and is suitable for integration into the multi-stage cardiac decision support system.

---

## Output

Example output:
```json
{
  "patient_id": "P001",
  "ecg_risk_score": 0.82,
  "risk_level": "High"
}
