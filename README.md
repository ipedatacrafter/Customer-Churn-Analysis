###Customer Churn Prediction & Visualization in Power BI###
##Tools Used: Python, Power BI, DAX, Pandas, XGBoost, Scikit-learn##

##Project Overview##
This project focuses on predicting customer churn probability using Python machine learning models and visualizing the insights through an interactive Power BI dashboard.

The main goal was to:

-Build a data-driven churn prediction model that identifies customers likely to leave.

-Provide business insights via interactive visuals and DAX-driven KPIs.

-Enable decision-makers to take proactive actions to reduce churn and improve retention.

-This project demonstrates my ability to integrate Machine Learning (Python) and Business Intelligence (Power BI) for real-world business solutions.

##Business Problem##
Customer churn directly impacts company revenue and growth. By identifying churn patterns early, businesses can implement targeted campaigns to retain valuable customers.

Key Business Questions:

-Which customers are at the highest risk of churning?

-What are the top features influencing churn behavior?

-How can business teams visualize churn insights for better decisions?
##Technical Approach##
##Step 1 — Data Cleaning & Preparation (Python)##
Handled missing values, feature engineering, and encoding.

df['tenure_days'] = (df['last_activity_date'] - df['signup_date']).dt.days
df['avg_order_value'] = df['revenue_last_3m'] / df['orders_last_3m'].replace(0, np.nan)
df['avg_order_value'] = df['avg_order_value'].fillna(0)
df = pd.get_dummies(df, columns=['region', 'product_category'], drop_first=True)

##Step 2 — Model Training##

Built and trained a predictive model using XGBoost for high accuracy and interpretability.

import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
model.fit(X_train, y_train)

##Step 3 — Model Evaluation##

Evaluated using AUC, precision, and recall.

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC Score:", auc)

##Step 4 — Export Predictions for Power BI##

test_out['predicted_proba'] = y_pred_proba
test_out['predicted_label'] = (y_pred_proba >= 0.5).astype(int)
test_out.to_csv("predictions_for_powerbi.csv", index=False)

##Power BI Measures (DAX Used)##

Total Customers = COUNTROWS(Customer)

Total Churn = 
CALCULATE(COUNTROWS(Customer), Customer[Churn] = "Yes")

Churn Rate = 
DIVIDE([Total Churn], [Total Customers], 0)

New Joiners = 
CALCULATE(COUNTROWS(Customer), Customer[New_Joiner] = "Yes")

Churn by Gender = 
CALCULATE([Total Churn], Customer[Gender])

Churn Rate by Contract = 
DIVIDE(
    CALCULATE(COUNTROWS(Customer), Customer[Churn] = "Yes"),
    CALCULATE(COUNTROWS(Customer))
)




