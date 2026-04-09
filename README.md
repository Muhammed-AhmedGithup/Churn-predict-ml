# Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company. Customer churn, the rate at which customers stop doing business with an entity, is a critical metric for businesses. Predicting churn allows companies to proactively engage at-risk customers and implement retention strategies.

## Project Overview
The goal of this project is to build a machine learning model that can accurately predict whether a customer will churn based on various service usage and demographic data. The project involves data loading, exploratory data analysis (EDA), data preprocessing, model training, and evaluation.

## Dataset
The dataset `train.csv` and `test.csv` contains information about customers, including:
- **Demographic Information**: Gender, SeniorCitizen, Partner, Dependents
- **Service Information**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account Information**: Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure
- **Target Variable**: Churn (Yes/No)

## Data Preprocessing
1.  **Missing Values**: Missing values were identified and handled. For this dataset, the `TotalCharges` column was dropped due to suspected data inconsistencies after initial analysis (though the original notebook shows it was dropped, a more robust approach might involve imputation or further investigation). Other columns with single missing values were implicitly handled by the LabelEncoder or dropped.
2.  **Duplicate Values**: The dataset was checked for duplicate rows.
3.  **Feature Engineering/Transformation**: Categorical features were converted into numerical representations using `LabelEncoder`. This was applied to all 'object' type columns in both training and test datasets.
4.  **Feature Scaling**: A `StandardScaler` was initialized for numerical features, though not explicitly applied in the final model pipeline in the provided code.

## Exploratory Data Analysis (EDA)
-   Distributions of key features like `gender`, `SeniorCitizen`, `MonthlyCharges` were visualized.
-   Churn rates were analyzed across different categorical variables to understand potential drivers of churn.

## Machine Learning Models
Two primary models were explored for churn prediction:

1.  **Logistic Regression**:
    -   A baseline `LogisticRegression` model was trained and evaluated.
    -   `accuracy_score`, `roc_auc_score`, and `confusion_matrix` were used for evaluation.
    -   An ROC curve was plotted to visualize model performance.

2.  **XGBoost Classifier (Optimized with RandomizedSearchCV)**:
    -   Recognizing the limitations of the simple `LogisticRegression` (especially with NaN handling), a more robust gradient boosting model (`XGBClassifier`) was chosen.
    -   `RandomizedSearchCV` with `StratifiedKFold` cross-validation was used to optimize hyperparameters and find the best model configuration based on `roc_auc` scoring.
    -   The best estimator from `RandomizedSearchCV` was selected as the final model.

## Prediction and Submission
The trained XGBoost model was used to predict churn probabilities on the unseen `test.csv` dataset. The predictions were then formatted into a submission file (`submission.csv`) containing `id` and `Churn` probabilities.

## Key Learnings/Challenges
-   Handling missing values effectively is crucial, especially when using models sensitive to NaNs like `LogisticRegression`.
-   `LabelEncoder` can introduce ordinality which might not be appropriate for all nominal categorical features, `OneHotEncoder` could be an alternative.
-   Cross-validation and hyperparameter tuning (`RandomizedSearchCV`) are essential for building robust and generalized models.
-   Understanding and addressing errors during model training (e.g., `LightGBMError: Number of classes must be 1 for non-multiclass training` and `ValueError: multi_class must be in ('ovo', 'ovr')` related to `y` having more than 2 unique values) is critical for successful model deployment.

## Setup and Usage
To run this notebook, ensure you have the following libraries installed:
```bash
pip install pandas scikit-learn seaborn matplotlib lightgbm xgboost
