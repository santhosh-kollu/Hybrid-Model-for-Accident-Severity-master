# Import necessary libraries
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import accuracy_score
import time

# Load the saved models
print("Loading saved models...")
start_time = time.time()
lgb_model = lgb.Booster(model_file='lgb_model.txt')
svm_model = joblib.load('svm_model.pkl')
rf_meta = joblib.load('rf_meta_model.pkl')
print(f"Models loaded in {time.time() - start_time:.2f} seconds")

# Load the balanced test data
print("Loading test data...")
X_test = pd.read_csv('balanced_X.csv')  # Full balanced data; we'll split later
y_test = pd.read_csv('balanced_y.csv').values.ravel()
X_test_ohe = np.load('balanced_X_ohe.npy')
y_test_ohe = np.load('balanced_y_ohe.npy')

# Simulate the train-test split to get the same test set (80:20 split, random_state=42)
print("Splitting data to match training split...")
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)
_, X_test_ohe, _, y_test_ohe = train_test_split(X_test_ohe, y_test_ohe, test_size=0.2, random_state=42)
print(f"Test set size: {X_test.shape[0]} samples")

# Generate predictions from base models
print("Generating LightGBM predictions...")
start_time_lgb = time.time()
lgb_test_pred = lgb_model.predict(X_test)
print(f"LightGBM predictions completed in {time.time() - start_time_lgb:.2f} seconds")

print("Generating SVM predictions...")
start_time_svm = time.time()
svm_test_pred = svm_model.predict_proba(X_test_ohe)
print(f"SVM predictions completed in {time.time() - start_time_svm:.2f} seconds")

# Stack predictions
print("Stacking predictions...")
stacked_test = np.hstack([lgb_test_pred, svm_test_pred])

# Generate final predictions with Random Forest meta-model
print("Generating final predictions with Random Forest...")
start_time_rf = time.time()
final_pred = rf_meta.predict(stacked_test)
print(f"Random Forest predictions completed in {time.time() - start_time_rf:.2f} seconds")

# Evaluate accuracy
accuracy = accuracy_score(y_test, final_pred)
print(f'Hybrid Model Test Accuracy: {accuracy:.4f}')
print(f"Total testing completed in {time.time() - start_time:.2f} seconds")