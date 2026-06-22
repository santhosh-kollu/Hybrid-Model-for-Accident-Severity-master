import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import lightgbm as lgb
import joblib  # For loading scikit-learn models
import matplotlib.pyplot as plt

# Load balanced datasets
print("Loading balanced datasets...")
X_balanced = pd.read_csv('balanced_X.csv')  # Features for LightGBM
y_balanced = pd.read_csv('balanced_y.csv').values.ravel()  # Target, flattened to 1D
X_ohe_balanced = np.load('balanced_X_ohe.npy')  # One-hot encoded features for SVM
y_ohe_balanced = np.load('balanced_y_ohe.npy')  # Target for SVM (assuming same as y_balanced)

# Verify SVM expected features
expected_features = X_ohe_balanced.shape[1]  # Number of features SVM expects (100)
print(f"SVM expects {expected_features} features.")

# Split into 90% training and 10% unseen test sets
X_train_full, X_unseen, y_train_full, y_unseen = train_test_split(X_balanced, y_balanced, test_size=0.1, random_state=42)
X_ohe_train_full, X_ohe_unseen, y_ohe_train_full, y_ohe_unseen = train_test_split(X_ohe_balanced, y_ohe_balanced, test_size=0.1, random_state=42)

# Further split the 90% training into 80% train and 20% mixed test (seen data)
X_train, X_seen, y_train, y_seen = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
X_ohe_train, X_ohe_seen, y_ohe_train, y_ohe_seen = train_test_split(X_ohe_train_full, y_ohe_train_full, test_size=0.2, random_state=42)

# Verify feature counts for seen data
print(f"X_seen has {X_seen.shape[1]} features for LightGBM.")
print(f"X_ohe_seen has {X_ohe_seen.shape[1]} features for SVM before alignment.")

# Align X_ohe_seen with training features (should already match, but ensure compatibility)
if X_ohe_seen.shape[1] != expected_features:
    X_ohe_seen_aligned = np.zeros((X_ohe_seen.shape[0], expected_features))
    min_features = min(X_ohe_seen.shape[1], expected_features)
    X_ohe_seen_aligned[:, :min_features] = X_ohe_seen[:, :min_features]
else:
    X_ohe_seen_aligned = X_ohe_seen  # No alignment needed if shapes match

# Load pretrained models
print("Loading pretrained models...")
lgb_model = lgb.Booster(model_file='lgb_model.txt')  # Load LightGBM model
svm_model = joblib.load('svm_model.pkl')  # Load SVM model
meta_model = joblib.load('rf_meta_model.pkl')  # Load Random Forest meta-model

# Evaluate on seen data (mixed test set)
print("Evaluating on seen data...")
lgb_seen_pred = lgb_model.predict(X_seen)  # LightGBM predictions
svm_seen_pred = svm_model.predict_proba(X_ohe_seen_aligned)  # SVM predictions
stacked_seen = np.hstack([lgb_seen_pred, svm_seen_pred])  # Stack predictions
y_seen_pred = meta_model.predict(stacked_seen)  # Meta-model prediction
seen_accuracy = accuracy_score(y_seen, y_seen_pred)
print(f"Accuracy on seen data: {seen_accuracy:.4f}")

# Generate confusion matrix for seen data
cm = confusion_matrix(y_seen, y_seen_pred)
print("Confusion Matrix for Seen Data:")
print(cm)

# Plot and save confusion matrix using ConfusionMatrixDisplay
plt.figure(figsize=(8, 6))
# Assuming y_balanced has string labels like 'Minor', 'Severe', 'Fatal'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_balanced))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Seen Data (Balanced Dataset)")
plt.savefig('confusion_matrix_seen.png', dpi=300, bbox_inches='tight')
plt.show()