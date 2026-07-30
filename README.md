# Cardio AI

> *"In cardiovascular care, time is muscle. Early detection and precise decision-support are not just technological advancements—they are essential lifelines."*

---
## Group Members  

| Person No. | IIT ID  | RGU ID  | Student Name                    |
|------------|--------|--------|--------------------------------|
| 1          | 20241269 |    2425798    | Mevin Samaranarayana           |
| 2          | 20240481 | 2425438 | Desindu Janandith Pannilage   |
| 3          | 20240875 |    2425460    | Chamath Hesara Gunawardhana    |
| 4          | 20241405 |   2425464     | Noor Mohammed Mohammed Fiham   |

---

## Introduction  


This project is an AI-Driven Coronary Disease Detection and Decision Support System designed to assist in early identification of cardiovascular risks using patient lifestyle data, ECG signals, and angiogram images. The system provides a multi-stage analysis pipeline to support clinicians and patients in understanding potential heart-related conditions.

To summarize here are the main features of the application:

- Initial risk screening using patient lifestyle data (Random Forest model)  
- 10-year coronary heart disease prediction using Logistic Regression model  
- Explainable AI (XAI)-based personalized recommendations  
- ECG images and Angiogram Videos Processing for analysis
- ECG image classification using CNN  
- Angiogram analysis using image processing and segmentation techniques  
- Blockage detection support using computational algorithms  
- Secure metadata-based storage system with encrypted file handling  
- Decision support system to assist clinical interpretation  

Although fairly advanced, the system is designed as a **decision support tool**, where final medical decisions are made by healthcare professionals.

---

## Features  

### Patient Portal  
Allows users to input lifestyle details such as BMI, glucose level, smoking habits, and medical history. The system performs initial risk assessment and provides recommendations using explainable AI.

### Current Risk Prediction System 
Implements an AI-driven initial risk screening system using Random Forest to identify high-risk cardiovascular patients based on clinical and lifestyle indicators.

### 10-year Risk Prediction System  
Integrates XGboosty and Logistic Regression models to predict both immediate risk and long-term (10-year) coronary heart disease probability.

### ECG Analysis Module  
Processes uploaded ECG images by converting them into structured signal data using image processing techniques and classifies them into risk categories (low, medium, high) using a CNN model.

### Angiogram Processing Module  
Extracts frames from angiogram videos, applies SSIM-based key frame selection, and enhances images using preprocessing techniques such as top-hat filtering.

### Vessel Segmentation & Blockage Detection  
Uses Deep Learning (DeepSA) for vessel segmentation and computational algorithms to identify vessel narrowing regions. These are highlighted as potential blockage areas for clinical review.

### Decision Support System  
Combines outputs from lifestyle data, ECG analysis, and angiogram processing to provide final risk assessment and recommendations for clinicians.

### Metadata & Storage System  
Implements a secure, session-based file storage system using encrypted files and structured JSON metadata for managing patient data efficiently.

These features collectively enhance the system’s ability to support early detection and assist clinical decision-making.

---

#  System Screenshots
<img width="954" height="631" alt="image" src="https://github.com/user-attachments/assets/ce4f17f6-c907-46e5-971f-7da3d2fbf03c" />
<img width="945" height="618" alt="image" src="https://github.com/user-attachments/assets/ff027cfa-262a-43e8-886f-b565fbb59876" />
<img width="838" height="1609" alt="image" src="https://github.com/user-attachments/assets/f56a06ac-3d77-4126-a200-e66e9aa3425d" />
<img width="931" height="805" alt="image" src="https://github.com/user-attachments/assets/8f6fb16b-ce27-4bb6-be11-be82d13c7266" />
<img width="945" height="849" alt="image" src="https://github.com/user-attachments/assets/ec68b7a4-94cb-4643-9e9b-421bf9e5987c" />
<img width="955" height="1308" alt="image" src="https://github.com/user-attachments/assets/5cd741ca-443e-475a-b198-7eb1db1ae740" />


## Project Management Methodology  
  

The project was managed using a collaborative and iterative development approach. GitHub was used for version control, where each team member maintained separate branches to manage their individual tasks and contributions. Regular discussions were conducted to track progress, resolve issues, and ensure smooth integration of system components. GitHub Projects were also used to organize tasks, set timelines, and monitor development activities throughout the project lifecycle.

---

## Folder Structure  

```plaintext
CM2603-DSGP-Group-3-main/
│
├── Dataset/                         # Raw datasets
├── EDA/                             # Exploratory Data Analysis
├── EDA Analysis/                    # Analysis outputs
├── Model/                           # Model development
├── Pipeline_Management/             # Encrypted Metadata Storage System
├── Preprocessed_Angiogram_Output/   # Processed angiogram outputs
├── Preprocessing/                   # ECG & Angiogram Image Processing
├── process_02_cardiac_analysis/     # Core analysis pipeline
├── static/                          # Frontend static files
├── templates/                       # HTML templates
├── ten_year_models/                 # 10 year risk models
├── uploads/                         # Uploaded ECG/angiogram files
│
├── app.py                           # Flask application
├── coronary_ai.db                   # Database
├── DSGP.ipynb                       # Experiments
└── .gitignore
```

---

## Pre-requisites  

Ensure the following are installed:

- Python 3.x  
- Required Python libraries  
- Virtual environment (recommended)  

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## What to run?  

To run the application execute below files respectively:

```bash
python demo.py
python app.py
```

---

## What to view?  

- If you wish to see the EDA carried out per each model look into the **EDA** folder  
- If you wish to run the application and view its contents use the **main project directory**  
- If you wish to see angiogram outputs refer to the **Preprocessed_Angiogram_Output** folder  
- If you wish to explore metadata and logs refer to the **Pipeline_Management** folder  

---

## Cloning the repository  

```bash
git clone https://github.com/MevinIIT/CM2603-DSGP-Group-3.git
```

---
