# -CSCI-MATH-485_Group-4-
# Loan Prediction System Using Machine Learning

## Project Overview
This project focuses on predicting loan approval status using Machine Learning techniques. Multiple classification models were developed and evaluated to identify the best-performing model for loan prediction.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Feature importance analysis
- SHAP explainability visualizations
- Web application deployment using Flask

---

## Team Information
Course: CSCI/MATH 485  
Project: Loan Approval Prediction Using Machine Learning
University: California State University, Chico

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SHAP
- Flask
- Jupyter Notebook

---

## Dataset Features
The dataset includes financial and demographic information such as:
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Property Area
- Education
- Employment Status
- Marital Status

Target Variable:
- Loan Approval Status

---

## Machine Learning Models Used
- Random Forest Classifier
- XGBoost Classifier
- Logistic Regression

---

## Project Structure

```bash
FinalProject/
│
├── app.py
├── templates/
│   └── index.html
├── loan_approval_dataset.csv
├── rf_loan_model.pkl
├── xgb_loan_model.pkl
├── xgboost_shap.ipynb
├── xgboost_shap_fixed.ipynb
├── visualization.ipynb
├── plots.ipynb
├── rf_confusion_matrix.png
├── rf_feature_importance.png
├── rf_prediction_distribution.png
├── rf_roc_curve.png
├── rf_shap_bar.png
├── rf_shap_summary.png
├── rf_shap_waterfall.png
└── README.md

Steps Performed
1. Data Preprocessing
Handled missing values
Encoded categorical variables
Feature scaling and transformation
2. Exploratory Data Analysis
Distribution analysis
Correlation analysis
Visualization of important features
3. Model Training
Trained Random Forest and XGBoost models
Evaluated using:
Accuracy
Confusion Matrix
ROC Curve
4. Explainable AI

SHAP (SHapley Additive exPlanations) was used to interpret model predictions and feature importance.

Visualizations included:
SHAP Summary Plot
SHAP Bar Plot
SHAP Waterfall Plot

Flask Web Application
The project includes a Flask-based web application for predicting loan approval using user input.

Run the application using:
python app.py

Open browser:
http://127.0.0.1:5000

Installation

Clone the repository:
git clone https://github.com/dokkaebi611/-CSCI-MATH-485_Group-4-.git

Move into project folder:
cd FinalProject

Install dependencies:
pip install -r requirements.txt

Results
Random Forest and XGBoost models achieved strong prediction accuracy.
SHAP explainability improved transparency and interpretability of predictions.
Flask deployment enabled interactive user prediction capability.
Future Improvements
Deploy application on cloud platforms
Improve model tuning
Add deep learning models
Enhance frontend UI/UX
Integrate database support

Conclusion
This project demonstrates the application of Machine Learning and Explainable AI techniques in loan approval prediction. The system provides accurate predictions while maintaining interpretability through SHAP-based analysis.
