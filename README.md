# Customer Churn Prediction 

## Project Overview

Customer Churn Prediction is a **machine learning–based project** that identifies customers who are likely to stop using a company’s services. Retaining customers is critical for business growth, and predicting churn in advance allows organizations to take proactive retention measures such as personalized offers, improved support, or targeted engagement strategies.

This project was developed as an **academic project** for the **Bachelor of Technology (Information Technology)** program at **KIIT Deemed to be University**.

---

## Objectives

* Predict whether a customer is likely to churn or not
* Analyze customer behavior and usage patterns
* Apply machine learning techniques for binary classification
* Evaluate model performance using standard metrics
* Provide actionable insights to reduce customer churn

---

## Key Concepts Used

* Customer Churn Analysis
* Supervised Machine Learning
* Binary Classification
* Feature Engineering & Selection
* Model Evaluation & Validation

---

## Tools & Technologies

* **Programming Language:** Python
* **Libraries Used:**

  * pandas
  * numpy
  * matplotlib
  * seaborn
  * scikit-learn

---

## Machine Learning Techniques

* **Logistic Regression** (Primary Model)
* Recursive Feature Elimination (RFE)
* Cross-Validation (K-Fold)

---

## Dataset

* Customer churn dataset (`new_churn_data.csv`)
* Contains demographic, behavioral, and service-related attributes
* Target variable: **Churn (Yes / No)**

---

## Methodology

1. **Data Collection**
   Customer data was collected from structured datasets containing usage and behavioral attributes.

2. **Data Preprocessing**

   * Handling missing values
   * Encoding categorical variables
   * Feature scaling and normalization
   * Removing irrelevant or redundant columns

3. **Exploratory Data Analysis (EDA)**

   * Distribution analysis
   * Correlation heatmaps
   * Visualization of churn vs non-churn customers

4. **Feature Selection**

   * Recursive Feature Elimination (RFE) used to select top features

5. **Model Training**

   * Logistic Regression model trained on processed data
   * Dataset split into training and testing sets

6. **Model Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion Matrix

---

## Results

* The model successfully identifies customers at risk of churning
* Achieved balanced performance across accuracy, precision, and recall
* Feature importance analysis revealed key factors influencing churn

---

## Project Structure

```
Churn-Prediction/
│
├── data/                  # Dataset files
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── model_training.py
│   └── evaluation.py
├── README.md
└── requirements.txt
```

---

## Future Scope

* Experiment with advanced models (Random Forest, XGBoost, Neural Networks)
* Real-time churn prediction system
* Integration with CRM platforms
* Explainable AI (SHAP / LIME) for interpretability
* Industry-specific churn models (telecom, banking, e-commerce)

---

## Project Team

* **Harsh Singh** (2106032)
* **Ayush Raj** (2106105)
* **Juned Eqbal** (2106124)
* **Megha Verma** (2106125)
* **Shaswat Kumar** (2106152)
* **Shruti Kumari** (2106155)

**Project Guide:**
Dr. Sarita Tripathy
School of Computer Engineering
KIIT Deemed to be University

---

## Conclusion

This project demonstrates how machine learning can be effectively used to predict customer churn and support data-driven business decisions. By identifying at-risk customers early, organizations can improve retention, customer satisfaction, and long-term profitability.

---

## References

* Python Official Documentation
* Scikit-learn Documentation
* Research papers o
