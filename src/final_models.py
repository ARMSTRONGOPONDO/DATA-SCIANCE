import os
import pandas as pd
import numpy as np
import logging
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, PrecisionRecallDisplay
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_imb_pipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from sklearn.utils.class_weight import compute_class_weight
import shap

# Configure logging and visualization
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. Data Loading with Path Validation
def load_data():
    """Load and validate dataset from working directory"""
    file_name = "realistic_kenyan_fraud_data.csv"
    file_path = os.path.join(os.getcwd(), file_name)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from: {file_path}")
        
        # Convert and validate columns
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['is_fraud'] = df['is_fraud'].astype(int)
        
        required_columns = {'transaction_id', 'user_name', 'credit_card_type', 
                           'transaction_amount', 'merchant_category', 'datetime',
                           'bank', 'location', 'is_foreign', 'transaction_type',
                           'transaction_frequency', 'time_since_last_txn_hrs', 'is_fraud'}
        
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing columns in dataset: {missing}")
            
        return df
    
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

# 2. Data Analysis and Visualization
def analyze_data(df):
    """Generate comprehensive visualizations of Kenyan transaction patterns"""
    plt.figure(figsize=(15, 20))
    
    # Fraud distribution by location
    plt.subplot(3, 2, 1)
    df[df['is_fraud'] == 1]['location'].value_counts().head(10).plot(kind='barh')
    plt.title('Top 10 Fraud Locations')
    
    # Transaction amount distribution
    plt.subplot(3, 2, 2)
    sns.histplot(df['transaction_amount'], bins=50, kde=True)
    plt.title('Transaction Amount Distribution')

    # Fraud rate by hour
    plt.subplot(3, 2, 3)
    df.groupby(df['datetime'].dt.hour)['is_fraud'].mean().plot()
    plt.title('Fraud Rate by Hour of Day')
    plt.xlabel('Hour of Day')
    
    # Transaction type distribution
    plt.subplot(3, 2, 4)
    df['transaction_type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Transaction Type Distribution')
    
    # Fraud correlation matrix
    plt.subplot(3, 2, 5)
    numeric_cols = ['transaction_amount', 'transaction_frequency', 
                   'time_since_last_txn_hrs', 'is_foreign', 'is_fraud']
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    
    # Fraud vs Normal transactions
    plt.subplot(3, 2, 6)
    sns.boxplot(x='is_fraud', y='transaction_amount', data=df)
    plt.title('Transaction Amount by Fraud Status')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('kenyan_fraud_analysis.png')
    plt.close()

# 2. Optimized Preprocessing Pipeline
def create_preprocessor():
    """Enhanced preprocessing for Kenyan transaction patterns"""
    categorical_features = [
        'credit_card_type', 'merchant_category', 
        'bank', 'transaction_type'
    ]
    
    numerical_features = [
        'transaction_amount', 'transaction_frequency',
        'time_since_last_txn_hrs', 'is_foreign'
    ]
    
    location_features = ['location']  # Special handling for Kenyan locations
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False), categorical_features),
            ('loc', OneHotEncoder(max_categories=20), location_features)
        ],
        remainder='drop'
    )
    
    return Pipeline([
        ('preprocessor', preprocessor)
    ])

# 3. Model Architectures with Complementary Strengths
def build_tensorflow_model(input_dim):
    """Deep model for complex pattern detection"""
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.6),
        Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc', curve='PR')
        ]
    )
    return model

def optimize_xgb():
    """High-precision model for rule-based patterns"""
    return XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        eval_metric='aucpr',
        tree_method='hist'
    )

def optimize_rf():
    """Conservative model for generalizable patterns"""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        class_weight='balanced_subsample',
        max_features='log2',
        n_jobs=-1
    )

# 4. Enhanced Training Workflow
def train_models(X_train, y_train):
    """Train complementary model ensemble"""
    # Convert to numpy arrays
    y_train = y_train.astype(int).to_numpy().flatten()
    
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    # Class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    
    # Neural Network
    nn_model = build_tensorflow_model(X_train_processed.shape[1])
    early_stopping = EarlyStopping(patience=10, monitor='val_auc', 
                                  restore_best_weights=True, mode='max')
    nn_model.fit(
        X_train_processed, y_train,
        epochs=100,
        batch_size=512,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2
    )
    nn_model.save('kenyan_fraud_nn.keras')
    
    # XGBoost with SMOTE
    xgb_pipe = make_imb_pipeline(
        SMOTE(sampling_strategy=0.5, k_neighbors=10),
        optimize_xgb()
    )
    xgb_pipe.fit(X_train_processed, y_train)
    joblib.dump(xgb_pipe, 'kenyan_fraud_xgb.pkl')
    
    # Random Forest
    rf_pipe = make_imb_pipeline(
        SMOTE(sampling_strategy=0.3),
        optimize_rf()
    )
    rf_pipe.fit(X_train_processed, y_train)
    joblib.dump(rf_pipe, 'kenyan_fraud_rf.pkl')
    
    return nn_model, xgb_pipe, rf_pipe

# 5. Comprehensive Model Analysis
def analyze_model(model, X_test, y_test, model_name, preprocessor):
    """In-depth model performance analysis"""
    X_test_processed = preprocessor.transform(X_test)
    
    # Predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test_processed)[:, 1]
    else:
        y_proba = model.predict(X_test_processed).flatten()
    
    # Metrics
    y_pred = (y_proba > 0.35).astype(int)
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.3f}")
    
    # Precision-Recall Curve
    disp = PrecisionRecallDisplay.from_predictions(y_test, y_proba, name=model_name)
    _ = disp.ax_.set_title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Feature Analysis
    if hasattr(model, 'feature_importances_'):
        if 'pipeline' in model.__class__.__name__.lower():
            importances = model.named_steps['xgbclassifier'].feature_importances_
        else:
            importances = model.feature_importances_
            
        feat_imp = pd.Series(importances, 
                            index=preprocessor.get_feature_names_out())
        plt.figure(figsize=(10,6))
        feat_imp.nlargest(15).plot(kind='barh')
        plt.title(f'{model_name} Feature Importances')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    # SHAP Explanations (sample 500 instances)
    sample_idx = np.random.choice(X_test_processed.shape[0], 500, replace=False)
    if any(m in model_name.lower() for m in ['xgb', 'randomforest']):
        try:
            if 'xgb' in model_name.lower():
                explainer = shap.TreeExplainer(model.named_steps['xgbclassifier'])
            else:
                explainer = shap.TreeExplainer(model.named_steps['randomforestclassifier'])
            
            shap_values = explainer.shap_values(X_test_processed[sample_idx])
            plt.figure()
            shap.summary_plot(shap_values, X_test_processed[sample_idx],
                             feature_names=preprocessor.get_feature_names_out(),
                             show=False)
            plt.title(f'{model_name} SHAP Summary')
            plt.tight_layout()
            plt.savefig(f'shap_{model_name.lower().replace(" ", "_")}.png')
            plt.close()
        except Exception as e:
            logger.error(f"SHAP failed for {model_name}: {str(e)}")

if __name__ == "__main__":
    try:
        # Load and validate data
        df = load_data()
        
        # Generate temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        
        # Perform comprehensive data analysis
        analyze_data(df)
        logger.info("Generated data visualizations: kenyan_fraud_analysis.png")
        
        # Prepare data for modeling
        X = df.drop(['is_fraud', 'transaction_id', 'user_name', 'datetime'], axis=1)
        y = df['is_fraud']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            stratify=y,
            random_state=42
        )
        
        # Train models
        nn_model, xgb_model, rf_model = train_models(X_train, y_train)
        preprocessor = joblib.load('preprocessor.pkl')
        
        # Evaluate models
        analyze_model(nn_model, X_test, y_test, "Neural Network", preprocessor)
        analyze_model(xgb_model, X_test, y_test, "XGBoost", preprocessor)
        analyze_model(rf_model, X_test, y_test, "Random Forest", preprocessor)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
