📘 **README.md — Telco Churn Prediction (Random Forest + Pipeline + Streamlit)**

Project Overview

This project builds a complete machine learning workflow for predicting customer churn using the Telco Customer Churn dataset. The solution includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit application for interactive predictions. The entire ML workflow is encapsulated inside a Scikit-Learn Pipeline, making the model fully reproducible and easy to deploy.

Goals

*   Create a robust and reliable churn prediction model
    
*   Use clean preprocessing with ColumnTransformer
    
*   Train a Random Forest classifier and tune it if necessary
    
*   Save the full pipeline for deployment
    
*   Provide a user-friendly Streamlit interface for predictions
    

Technologies Used

*   Python 3.9+
    
*   Scikit-Learn
    
*   Pandas
    
*   NumPy
    
*   Matplotlib
    
*   Joblib
    
*   Streamlit
    

Project Structure

├── dataset Telco-Customer-Churn.csv
│
├── churn analysis.ipynb # training & evaluation notebook
│
├── app.py # Streamlit application
├── best_model.joblib # trained pipeline (preprocessing + model)
│
├── README.md
└── requirements.txt

Dataset Description

The dataset includes customer demographic data, service subscription info, monthly charges, contract type, and churn status.

Main preprocessing steps:

*   Remove the column “customerID,” which is not predictive
    
*   Convert TotalCharges to numeric values
    
*   Map Churn to numerical labels (No = 0, Yes = 1)
    
*   Split features into numerical and categorical groups
    
*   Handle missing values
    
*   Encode categorical variables using OneHotEncoder
    
*   Scale numeric variables using StandardScaler
    

Modeling

A Random Forest classifier is used as the main model due to its robustness, ability to handle non-linear relationships, and strong performance with mixed data types. The model is integrated into a Scikit-Learn Pipeline that includes all preprocessing steps.

Evaluation

The model is evaluated using metrics suitable for churn prediction:

*   ROC AUC
    
*   Precision
    
*   Recall
    
*   F1-score
    
*   Confusion matrix
    
*   Precision-Recall curveAdditionally, the decision threshold can be customized to balance between recall and precision depending on business needs.
    

Hyperparameter Tuning

GridSearchCV is optionally used with stratified cross-validation and ROC AUC scoring to find optimized Random Forest configurations.

Model Saving

The trained pipeline (including preprocessing and the Random Forest classifier) is saved using joblib:

joblib.dump(best\_model, "best\_model.joblib")

Deployment with Streamlit

A Streamlit application (app.py) is included, allowing users to enter customer attributes or upload a CSV file for batch predictions.

Features of the app:

*   Single customer prediction
    
*   Probability of churn
    
*   Adjustable classification threshold
    
*   Batch prediction using CSV files
    
*   Automatic compatibility with the full preprocessing pipeline
    

Installation and Usage

Install dependencies:

pip install -r requirements.txt

Run the app:

streamlit run app.py

Future Improvements

*   Add SHAP or permutation feature importance for explainability
    
*   Create a FastAPI service for production inference
    
*   Add data drift monitoring
    
*   Experiment with more advanced algorithms such as XGBoost or LightGBM
    
*   Deploy on Streamlit Cloud or AWS
    

Author

Project developed by Leonardo Said as part of machine learning and model deployment studies.
