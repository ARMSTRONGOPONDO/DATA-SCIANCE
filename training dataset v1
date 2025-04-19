import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from xgboost import XGBClassifier
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
import joblib
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Step 1: Data Loading and Exploration
def load_data(files):
    """Load and concatenate multiple CSV files."""
    dataframes = [pd.read_csv(file) for file in files]
    combined_data = pd.concat(dataframes, ignore_index=True)
    return combined_data

# Step 2: Data Cleaning and Preprocessing
def preprocess_data(df):
    logger.info("Starting data preprocessing.")

    # Handle missing values
    logger.info("Handling missing values.")
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('missing')
        else:
            df[col] = df[col].fillna(df[col].median())

    # Clean column names for consistency
    logger.info("Cleaning column names.")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Handle binary labels with inconsistent naming
    if 'fraud' in df.columns:
        logger.info("Standardizing target column 'fraud'.")
        df['fraud'] = df['fraud'].replace({"is_fraud": 1, "fraud": 1, "Fraud": 0, 1: 1, 1.0: 1, 0.0: 0})

    if 'class' in df.columns:
        logger.info("Handling binary labels in 'Class' column.")
        df['fraud'] = df['class'].replace({0: 0, 1: 1})

    # Drop columns that could cause data leakage
    logger.info("Checking and dropping leakage columns.")
    leakage_cols = ['class', 'is_fraud', 'Fraud']  # Specify all potential leakage columns except 'fraud'
    for col in leakage_cols:
        if col in df.columns:
            logger.warning(f"Dropping potential leakage column: {col}")
            df.drop(col, axis=1, inplace=True)

    # Extract temporal features if date/time columns exist
    if 'date' in df.columns:
        logger.info("Extracting temporal features from the 'date' column.")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df.drop('date', axis=1, inplace=True)

    # Separate features and target
    if 'fraud' in df.columns:
        logger.info("Separating features and target.")
        y = df['fraud']
        X = df.drop('fraud', axis=1)
    else:
        logger.error("Target column 'fraud' not found in the dataset.")
        raise ValueError("Target column 'fraud' not found in the dataset.")

    logger.info("Data preprocessing completed.")
    return X, y

# Step 3: Feature Engineering
def feature_engineering(X):
    """Apply encoding to categorical features and scale numerical features."""
    
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    column_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return column_transformer

# Step 4: Model Development - TensorFlow Model
def build_tensorflow_model(input_dim):
    """Build a TensorFlow Sequential model."""
    model = Sequential([
        Input(shape=(input_dim,)),  # Define the input shape explicitly
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Step 5: Train and Save Models
def train_and_save_models(X_train, y_train):
    """Train TensorFlow, XGBoost, and meta-models, and save them to disk."""

    # Feature Engineering
    column_transformer = feature_engineering(X_train)
    X_train_transformed = column_transformer.fit_transform(X_train)

    # Save the fitted ColumnTransformer
    joblib.dump(column_transformer, "column_transformer.pkl")

    # Train TensorFlow model
    input_dim = X_train_transformed.shape[1]
    tf_model = build_tensorflow_model(input_dim)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    tf_model.fit(
        X_train_transformed, y_train,
        validation_split=0.2, epochs=50, batch_size=32, callbacks=[early_stopping]
    )
    tf_model.save("tensorflow_model.keras")

    # Train XGBoost model
    xgb_model = XGBClassifier(eval_metric='logloss')
    xgb_model.fit(X_train_transformed, y_train)
    joblib.dump(xgb_model, "xgboost_model.pkl")  # Save with joblib

    # Meta-model (Logistic Regression)
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_transformed, y_train)
    joblib.dump(rf_model, "meta_model.pkl")

# Step 6: Model Evaluation
def evaluate_models(X_test, y_test):
    """Evaluate TensorFlow, XGBoost, and meta-models."""

    # Load the fitted ColumnTransformer
    column_transformer = joblib.load("column_transformer.pkl")
    X_test_transformed = column_transformer.transform(X_test)

    # Load models
    tf_model = tf.keras.models.load_model("tensorflow_model.keras")
    xgb_model = joblib.load("xgboost_model.pkl")  # Load with joblib
    rf_model = joblib.load("meta_model.pkl")

    # Make predictions
    tf_preds = (tf_model.predict(X_test_transformed) > 0.5).astype(int)
    xgb_preds = xgb_model.predict(X_test_transformed)
    rf_preds = rf_model.predict(X_test_transformed)

    # Evaluate each model
    for model_name, preds in zip(['TensorFlow', 'XGBoost', 'Meta-Model'], [tf_preds, xgb_preds, rf_preds]):
        print(f"\n{model_name} Classification Report:\n")
        print(classification_report(y_test, preds))
        print("ROC AUC Score:", roc_auc_score(y_test, preds))

if __name__ == "__main__":
    files = ["/home/armstrong/card_transdata (original).csv", "/home/armstrong/creditcard.csv", "/home/armstrong/creditcard_2023.csv", "/home/armstrong/credit_card_transactions.csv"]
    df = load_data(files)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_and_save_models(X_train, y_train)
    evaluate_models(X_test, y_test)
