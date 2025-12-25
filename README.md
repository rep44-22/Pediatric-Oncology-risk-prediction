📌 Project Overview

This project focuses on risk stratification by predicting the probability of death for patients using clinical and demographic data. 
The model outputs a continuous death probability rate, enabling nuanced risk assessment rather than a binary outcome. This approach supports early identification of high-risk patients and better-informed decision-making.

📊 Dataset Description
The dataset contains patient-level clinical information, including demographic attributes, cancer-related variables, and outcome indicators. The death probability rate is used as the target variable, while the binary death label is excluded to prevent data leakage.

⚙️ Data Preprocessing
The data undergoes multiple preprocessing steps to ensure high-quality model input:
Feature–target separation with death probability rate as the prediction target
Encoding of categorical variables such as gender and cancer type
Standardization of time-based features such as years since diagnosis
These steps ensure consistent feature scaling and improve model stability.

🔀 Model Training and Validation
The dataset is split into training and testing sets using an 80–20 ratio. A Random Forest regression model is employed to learn complex, non-linear relationships between patient features and mortality risk.
Model performance is optimized using randomized hyperparameter tuning with cross-validation, balancing accuracy and computational efficiency.

📈 Model Output and Evaluation
The model produces a predicted death probability rate for each patient, expressed as a continuous value. Performance is evaluated on unseen data using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
These metrics assess how closely the predicted probability rates align with actual outcomes.

🔎 Model Explainability
To ensure transparency, SHAP-based explainability is applied:
Global feature importance identifies key risk contributors
Feature impact analysis explains how individual patient attributes increase or decrease the predicted death probability rate
This makes the model suitable for high-stakes, healthcare-related applications.
