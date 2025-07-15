### Predicting-Modelling-For-Liver-Disease-Detection
This project uses machine learning to predict liver disease based on patient health data. After testing 8 models, Gradient Boosting gave the best results with 92% accuracy and 96% recall. The model is saved and can predict liver disease for new patients using basic inputs

#### 🚀 Project Overview

This project develops a machine learning tool to predict liver disease using common patient health data. Early-stage liver disease often lacks clear symptoms and is difficult to detect, especially in regions with limited medical access. Our model leverages clinical and lifestyle information—like age, alcohol use, BMI, and blood test results—to predict the presence of liver disease, allowing for faster diagnosis, better treatment, and improved patient care.
t early symptoms. Our goal is to:
- Detect liver disease early using easily available medical data.
- Support clinical decision-making.
- Reduce the need for costly tests and hospital visits.
- Provide healthcare support in remote or resource-limited areas.

#### 📊 Dataset

- **Source**: [Kaggle - Predict Liver Disease Dataset (1700 Records)](https://www.kaggle.com/datasets/rabieelkharoua/predict-liver-disease-1700-records-dataset/data)
- **Type**: Classification
- **Size**: 1700 records, multiple clinical and lifestyle features.

#### Key Features Used:
- Age, Gender, BMI  
- Alcohol Consumption, Smoking Status  
- Genetic Risk, Physical Activity  
- Diabetes, Hypertension  
- Liver Function Test Results  

---

#### ⚙️ Project Workflow

1. **Data Cleaning & Feature Selection**
2. **Feature Scaling on Numerical Variables**
3. **Model Training with 8 Classifiers**
4. **Performance Evaluation on Metrics (Accuracy, Recall, F1-score)**
5. **Model Comparison and Selection**
6. **Prediction on New Patient Input**
7. **Result Output with Prediction + Probability**

---

#### 🧪 Models Trained

- Logistic Regression  
- K-Nearest Neighbors  
- Decision Tree  
- Random Forest  
- Support Vector Machine  
- Naive Bayes  
- XGBoost  
- **Gradient Boosting (Best Performer)**

---

#### 🏆 Best Model Performance (Gradient Boosting)

- **Test Accuracy:** 92.06%  
- **Test Recall:** 96.07%  
- **Test F1-Score:** 92.68%  

> Gradient Boosting was the most balanced and reliable model, especially suitable for medical use cases where high **recall** is critical.

---

#### 🛠️ Tech Stack

- **Programming Language**: Python  
- **Libraries**:  
  `pandas`, `numpy`, `matplotlib`, `seaborn`,  
  `scikit-learn`, `xgboost`, `joblib`

- **Tools**: Jupyter Notebook

---

#### 📦 Use Cases

- Early-stage liver disease detection  
- Clinical decision support tool  
- Integration into health tech apps  
- Patient self-assessment for awareness

---

#### ✅ Final Conclusion

This project proves how machine learning can assist healthcare by providing accurate and fast risk predictions for liver disease. Our Gradient Boosting model helps detect liver disease earlier, leading to better treatment, lower costs, and healthier outcomes.

---
