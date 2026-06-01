# ❤️ Cardiovascular Disease Prediction using Machine Learning

## 📌 Overview

Cardiovascular diseases (CVDs) are among the leading causes of death worldwide. Early identification of high-risk patients can help improve preventive care and treatment outcomes.

This project uses Machine Learning techniques to predict the likelihood of cardiovascular disease based on patient health indicators such as age, blood pressure, cholesterol levels, glucose levels, BMI, smoking habits, alcohol consumption, and physical activity.

The final model was built using **XGBoost**, which achieved the best performance among all evaluated models.

---

## 🚀 Live Features

* Patient Risk Prediction
* XGBoost-Based Classification Model
* Automated Feature Engineering
* Interactive Streamlit Web Application
* Risk Probability Visualization
* Feature Importance Analysis

---

## 📊 Dataset

**Source:** Kaggle Cardiovascular Disease Dataset

* Total Records: 70,000
* Cleaned Records: 68,702
* Target Variable: `cardio`
* Problem Type: Binary Classification

### Features

| Feature     | Description              |
| ----------- | ------------------------ |
| age         | Patient Age              |
| gender      | Gender                   |
| height      | Height (cm)              |
| weight      | Weight (kg)              |
| ap_hi       | Systolic Blood Pressure  |
| ap_lo       | Diastolic Blood Pressure |
| cholesterol | Cholesterol Level        |
| gluc        | Glucose Level            |
| smoke       | Smoking Status           |
| alco        | Alcohol Consumption      |
| active      | Physical Activity        |
| cardio      | Target Variable          |

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

* Dataset Inspection
* Missing Value Analysis
* Statistical Summary
* Target Distribution Analysis
* Correlation Heatmap
* Outlier Detection using Boxplots

### Outlier Handling

Removed unrealistic values from:

* Height
* Weight
* Systolic Blood Pressure
* Diastolic Blood Pressure

---

## ⚙️ Feature Engineering

To improve model performance, the following features were created:

### BMI (Body Mass Index)

BMI = Weight / Height²

### Pulse Pressure

Pulse Pressure = Systolic BP − Diastolic BP

### Mean Arterial Pressure (MAP)

MAP = (2 × Diastolic BP + Systolic BP) / 3

### Age Conversion

Converted age from days to years.

---

## 🤖 Models Evaluated

The following machine learning models were trained and compared:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* XGBoost

---

## 📈 Model Performance

| Model               |   Accuracy |
| ------------------- | ---------: |
| XGBoost             | **73.17%** |
| Random Forest       |     73.14% |
| SVM                 |     72.90% |
| Decision Tree       |     72.61% |
| Logistic Regression |     72.28% |
| KNN                 |     70.50% |

---

## 🏆 Final Model

### XGBoost Classifier

Parameters:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)
```

---

## 📋 Evaluation Metrics

| Metric        | Value  |
| ------------- | ------ |
| Accuracy      | 73.17% |
| ROC-AUC Score | 0.80   |

### Why ROC-AUC?

For medical classification problems, ROC-AUC is often more informative than accuracy because it evaluates how well the model distinguishes between patients with and without cardiovascular disease across different thresholds.

---

## 📊 Feature Importance

Top features identified by XGBoost:

| Feature     | Importance |
| ----------- | ---------: |
| ap_hi       |      0.629 |
| cholesterol |      0.099 |
| map         |      0.075 |
| age         |      0.060 |
| active      |      0.025 |
| smoke       |      0.020 |
| alco        |      0.016 |
| bmi         |      0.016 |
| gluc        |      0.015 |
| weight      |      0.012 |

### Key Finding

Systolic Blood Pressure (`ap_hi`) was the strongest predictor of cardiovascular disease, followed by cholesterol levels, MAP, and age.

---

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Joblib
* Streamlit

---

## 🌐 Deployment

The trained model was saved using Joblib:

```python
joblib.dump(xgb, "best_cardio_xgboost_model.pkl")
```

The application was deployed using Streamlit for real-time cardiovascular risk prediction.

---

## 📁 Project Structure

```text
cardiovascular_prediction/
│
├── app.py
├── best_cardio_xgboost_model.pkl
├── requirements.txt
├── notebook.ipynb
├── cardio_train.csv
└── README.md
```

---

## 📌 Conclusion

This project demonstrates an end-to-end Machine Learning pipeline for cardiovascular disease prediction, including:

* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Model Comparison
* XGBoost Training
* Model Evaluation
* Streamlit Deployment

The final XGBoost model achieved **73.17% Accuracy** and **0.80 ROC-AUC**, making it effective for identifying individuals at elevated cardiovascular risk.

---

### 👨‍💻 Author

**Yash Pratap Rai**

Machine Learning | Data Science | AI Enthusiast
