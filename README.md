<div align="center">

# ❤️ Cardio Risk Predictor

### Advanced Cardiovascular Disease Risk Assessment System

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://cardiovascularprediction-nvluftnarc9lv2eapp5asf.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

> **Predict cardiovascular disease risk using 14 engineered clinical parameters — powered by a Gradient Boosting model trained on 70,000 patient records.**

<br/>

![Accuracy](https://img.shields.io/badge/Accuracy-73.31%25-brightgreen?style=flat-square)
![ROC AUC](https://img.shields.io/badge/ROC_AUC-80.04%25-blue?style=flat-square)
![Training Records](https://img.shields.io/badge/Training_Records-70K-purple?style=flat-square)
![Features](https://img.shields.io/badge/Features-14_Clinical-orange?style=flat-square)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Input Features](#-input-features)
- [Getting Started](#-getting-started)
- [App Sections](#-app-sections)
- [Risk Interpretation](#-risk-interpretation)
- [Disclaimer](#-disclaimer)
- [Author](#-author)

---

## 🎯 Overview

**Cardio Risk Predictor** is a full-stack machine learning web application that estimates a patient's probability of having cardiovascular disease based on clinical measurements and lifestyle factors. The system uses a trained **Gradient Boosting Classifier** and provides real-time predictions through an interactive, beautifully designed Streamlit interface.

This project covers the full ML lifecycle — data preprocessing, feature engineering, model training & validation, and deployment as a production-ready web app.

---

## 🚀 Live Demo

👉 **[Try it Live on Streamlit](https://cardiovascularprediction-nvluftnarc9lv2eapp5asf.streamlit.app/)**

No installation required — run it directly in your browser!

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🩺 **Risk Prediction** | Real-time cardiovascular risk assessment with probability score |
| 📊 **Model Dashboard** | Confusion Matrix, ROC Curve, and Precision-Recall Curve |
| 📁 **Batch Prediction** | Upload a CSV and process hundreds of patient records at once |
| 📈 **Interactive Visuals** | Plotly-powered charts, donut risk gauge, and blood pressure indicator |
| 🔢 **Auto-computed Features** | BMI, Pulse Pressure, and SBP/DBP Ratio calculated on-the-fly |
| 💾 **Export Results** | Download batch predictions as a CSV file |
| 📱 **Responsive UI** | Clean, modern interface with color-coded risk levels |

---

## 📊 Model Performance

The final model was evaluated on a held-out test set:

| Metric | Score |
|---|---|
| ✅ Accuracy | **73.31%** |
| 📈 ROC AUC | **80.04%** |
| 🔁 Validation | 5-Fold Cross-Validation |
| 🗃️ Training Data | 70,000 patient records |

**Risk Level Interpretation:**

```
< 30%   →  🟢 Very Low Risk
30–49%  →  🟡 Low Risk
50–69%  →  🟠 Moderate Risk
70–89%  →  🔴 High Risk
≥ 90%   →  🚨 Very High Risk
```

---

## 🛠️ Tech Stack

```
Backend / ML          Frontend / UI         Data Processing
─────────────────     ─────────────────     ─────────────────
Python 3.8+           Streamlit             Pandas
Scikit-learn 1.5.0    Plotly                NumPy
XGBoost               Matplotlib            Joblib
imbalanced-learn      Seaborn
```

---

## 📁 Project Structure

```
cardiovascular_prediction/
│
├── app.py                          # Main Streamlit application (777 lines)
├── model.ipynb                     # Model training & evaluation notebook
├── best_cardio_model_final.joblib  # Trained Gradient Boosting model
│
├── cardio_train.csv                # Original training dataset
├── X_test.csv                      # Test features (for performance dashboard)
├── y_test.csv                      # Test labels
├── y_pred.csv                      # Model predictions on test set
│
├── requirements.txt                # Python dependencies
└── .devcontainer/                  # Dev container configuration
```

---

## 🔬 Input Features

The model uses **14 engineered clinical parameters**:

### Direct Inputs
| Feature | Description | Type |
|---|---|---|
| `age_years` | Patient age | Numeric |
| `height` | Height in cm | Numeric |
| `weight` | Weight in kg | Numeric |
| `ap_hi` | Systolic blood pressure (mmHg) | Numeric |
| `ap_lo` | Diastolic blood pressure (mmHg) | Numeric |
| `gender` | 1 = Male, 2 = Female | Categorical |
| `cholesterol` | 1 = Normal, 2 = Above Normal, 3 = High | Categorical |
| `gluc` | 1 = Normal, 2 = Above Normal, 3 = High | Categorical |
| `smoke` | 0 = Non-Smoker, 1 = Smoker | Categorical |
| `alco` | 0 = No, 1 = Yes | Categorical |
| `active` | 0 = Inactive, 1 = Active | Categorical |

### Auto-Computed Features
| Feature | Formula |
|---|---|
| `bmi` | `weight / (height/100)²` |
| `pulse_pressure` | `ap_hi - ap_lo` |
| `sbp_dbp_ratio` | `ap_hi / ap_lo` |

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YashPratapRai/cardiovascular_prediction.git
cd cardiovascular_prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

> **Note:** Ensure `best_cardio_model_final.joblib`, `X_test.csv`, and `y_test.csv` are in the same directory as `app.py` for full functionality.

---

## 📱 App Sections

### 🏠 Predict
Enter patient details across three tabs — **Basic Information**, **Vital Signs**, and **Lifestyle Factors** — then click **Predict** to instantly get:
- A visual risk donut chart with percentage
- Color-coded risk classification (High Risk / Low Risk)
- Key contributing health metrics

### 📊 Model Performance
View the model's evaluation metrics on the test dataset:
- Confusion Matrix (heatmap)
- ROC Curve with AUC score
- Precision-Recall Curve

### 📁 Batch Prediction
Upload a `.csv` file containing multiple patient records. The app will:
- Validate required columns
- Compute derived features automatically
- Predict risk for every patient
- Display a styled, color-gradient results table
- Offer a timestamped CSV download

### ℹ️ About / Info
Overview of the project, model details, feature importance chart, and developer contact.

---

## ⚠️ Disclaimer

> This tool is intended for **educational and informational purposes only**. It is **not a substitute for professional medical advice, diagnosis, or treatment**. Always consult a qualified healthcare provider for any medical concerns.

---

## 👨‍💻 Author

**Yash Pratap Rai**

[![GitHub](https://img.shields.io/badge/GitHub-YashPratapRai-181717?style=for-the-badge&logo=github)](https://github.com/YashPratapRai)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-yash--pratap--rai-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/yash-pratap-rai/)
[![Email](https://img.shields.io/badge/Email-raiyashpratap@gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:raiyashpratap@gmail.com)

---

<div align="center">

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ by [Yash Pratap Rai](https://github.com/YashPratapRai)

</div>
