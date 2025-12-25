import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
data=pd.read_csv("pediatric_mortality_clean.csv")
print(data.head(10))
X=data.drop(columns=['death','death_probability'])
y=data['death_probability']
encoder=LabelEncoder()
X['gender']=encoder.fit_transform(X['gender'])
X['cancer_type']=encoder.fit_transform(X['cancer_type'])


scaler=StandardScaler()
spec_cols=['years_since_diagnosis']

scaler.fit(X[spec_cols])
X[spec_cols]=scaler.transform(X[spec_cols])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
param_dist = {
    'n_estimators': randint(100, 400),
    'max_depth': [10, 15, 20, None],
    'min_samples_split': randint(2, 6),
    'min_samples_leaf': randint(1, 3),
    'max_features': ['sqrt', 'log2']
}
rf = RandomForestRegressor(random_state=42)

grid_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,              # only 20 random combos instead of hundreds
    cv=3,                   # 3-fold CV instead of 5
    scoring='neg_mean_absolute_error',
    #n_jobs=-1,
    random_state=42,
    verbose=1
)


grid_search.fit(X_train, y_train)
y_pred=grid_search.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
explainer = shap.Explainer(grid_search.best_estimator_)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar") # Bar plot for overall feature importance
print()
shap.summary_plot(shap_values, X_test) # Beeswarm plot for feature impact and value
