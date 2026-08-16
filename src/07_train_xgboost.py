import pandas as pd
import numpy as np
import os
import joblib
import category_encoders as ce
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score
import xgboost as xgb

input_file = './data/processed/advanced_features_600K.csv'
model_dir = './models'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print("Loading engineered features...")
df = pd.read_csv(input_file, low_memory=False)

target_col = 'ArrDel15'

drop_cols = ['FlightDate', target_col]
drop_cols = [c for c in drop_cols if c in df.columns]

X = df.drop(columns=drop_cols)
y = df[target_col]

categorical_cols = ['Reporting_Airline', 'Origin', 'Dest', 'ROUTE']

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

print("Applying target encoding...")
target_encoder = ce.TargetEncoder(cols=categorical_cols)

# Fit and transform on training data ONLY to prevent leakage
X_train_encoded = target_encoder.fit_transform(X_train, y_train)
# Transform test data based on training mappings
X_test_encoded = target_encoder.transform(X_test)

scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

print("Initializing GridSearchCV for hyperparameter tuning...")
base_model = xgb.XGBClassifier(
    scale_pos_weight=scale_weight,
    eval_metric="auc",
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [300, 500],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='recall',  # strictly fights for maximum Recall
    cv=3,
    verbose=2,
    n_jobs=-1
)

print("Running Grid Search (Optimizing for Recall)...")
grid_search.fit(X_train_encoded, y_train)

print(f"Best Parameters Found: {grid_search.best_params_}")
model = grid_search.best_estimator_

print("Generating predictions and evaluating best model...")
y_pred = model.predict(X_test_encoded)
y_prob = model.predict_proba(X_test_encoded)[:, 1]

print("\nClassification Report (Default 50% Threshold):")
print(classification_report(y_test, y_pred))

auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.4f}")

print(" ADVANCED: THRESHOLD TUNING FOR MAX RECALL")

# Sweeping thresholds to find the sweet spot for business operations
thresholds = [0.50, 0.45, 0.40, 0.35]
for t in thresholds:
    custom_preds = (y_prob >= t).astype(int)
    r = recall_score(y_test, custom_preds)
    p = precision_score(y_test, custom_preds)
    print(f"Classification probability Threshold {int(t*100)}% -> Precision: {p:.2f} | Recall: {r:.2f}")

print("\nSaving model and artifacts...")
joblib.dump(model, os.path.join(model_dir, 'xgboost_model.pkl'))
joblib.dump(target_encoder, os.path.join(model_dir, 'target_encoder.pkl'))
joblib.dump(X_train_encoded.columns.tolist(), os.path.join(model_dir, 'feature_columns.pkl'))
print(" Done! Your High-Recall model is saved and ready for the Dashboard.")