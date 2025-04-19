Fraud Analysis Insights based on the outcome and the interpretation of the visual outputs
(remember the data is synthetic it is not statistically accurate)

Top Fraud Locations: Kericho, Kilifi, and Embu are the top regions with high fraud activity.
Transaction Patterns:

ATM transactions dominate (34.9%), followed by Retail (33.7%).

Higher transaction_amount and transaction_frequency strongly correlate with fraud (0.76).

Fraudulent transactions often occur shortly after the last transaction (time_since_last_txn_hrs correlation: -0.4).

Model Performance:

Neural Network achieved the highest AP (0.94), slightly outperforming Random Forest and XGBoost (both AP = 0.93).

SHAP Analysis (XGBoost): Key fraud indicators include transaction_amount, transaction_frequency, and merchant categories like Tech Africa and Online Forex.

Visualizations:

Precision-Recall curves and feature correlation matrices are included for model evaluation.
