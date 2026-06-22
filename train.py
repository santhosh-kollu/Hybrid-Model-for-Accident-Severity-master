# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from tqdm import tqdm
import time
import joblib  # For saving scikit-learn models
from lime.lime_tabular import LimeTabularExplainer  # Import LIME

# Load balanced datasets
print("Loading balanced datasets...")
start_time = time.time()
X_balanced = pd.read_csv('balanced_X.csv')
y_balanced = pd.read_csv('balanced_y.csv').values.ravel()  # Convert to 1D array
X_ohe_balanced = np.load('balanced_X_ohe.npy')
y_ohe_balanced = np.load('balanced_y_ohe.npy')

# Split into training and testing sets
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
X_train_ohe, X_test_ohe, y_train_ohe, y_test_ohe = train_test_split(X_ohe_balanced, y_ohe_balanced, test_size=0.2, random_state=42)

# --- Base Model 1: LightGBM (on one-hot encoded data) ---
lgb_params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_balanced)),
    'verbose': 100,  # Show progress every 100 iterations
    'early_stopping_rounds': 50  # Stop if no improvement for 50 rounds
}
print("Training LightGBM model on one-hot encoded data...")
start_time_lgb = time.time()

# Prepare LightGBM dataset (using one-hot encoded data)
lgb_train = lgb.Dataset(X_train_ohe, label=y_train)
lgb_test = lgb.Dataset(X_test_ohe, label=y_test, reference=lgb_train)

# Cross-validation predictions with tqdm
cv = 5
lgb_oof = np.zeros((X_train_ohe.shape[0], len(np.unique(y_balanced))))
fold_size = len(X_train_ohe) // cv
for i in tqdm(range(cv), desc="LightGBM CV"):
    start_idx = i * fold_size
    end_idx = (i + 1) * fold_size if i < cv - 1 else len(X_train_ohe)
    train_idx = np.arange(start_idx, end_idx)
    val_idx = np.array([j for j in range(len(X_train_ohe)) if j not in train_idx])
    lgb_cv_train = lgb.Dataset(X_train_ohe[train_idx], y_train[train_idx])
    lgb_cv_val = lgb.Dataset(X_train_ohe[val_idx], y_train[val_idx], reference=lgb_cv_train)
    lgb_model_cv = lgb.train(lgb_params, lgb_cv_train, num_boost_round=1000, valid_sets=[lgb_cv_val])
    lgb_oof[val_idx] = lgb_model_cv.predict(X_train_ohe[val_idx])

# Final training
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=1000, valid_sets=[lgb_test])
lgb_test_pred = lgb_model.predict(X_test_ohe)
print(f"LightGBM training completed in {time.time() - start_time_lgb:.2f} seconds")

# Save LightGBM model
print("Saving LightGBM model...")
lgb_model.save_model('lgb_model.txt')

# --- Base Model 2: SVM ---
svm_model = SVC(probability=True, verbose=True)
print("Training SVM model...")
start_time_svm = time.time()
svm_oof = cross_val_predict(svm_model, X_train_ohe, y_train_ohe, cv=5, method='predict_proba', n_jobs=-1)
svm_model.fit(X_train_ohe, y_train_ohe)
svm_test_pred = svm_model.predict_proba(X_test_ohe)
print(f"SVM training completed in {time.time() - start_time_svm:.2f} seconds")

# Save SVM model
print("Saving SVM model...")
joblib.dump(svm_model, 'svm_model.pkl')

# --- Stack Predictions ---
print("Stacking predictions...")
stacked_oof = np.hstack([lgb_oof, svm_oof])
stacked_test = np.hstack([lgb_test_pred, svm_test_pred])

# --- Meta-Model: Random Forest ---
rf_meta = RandomForestClassifier(n_estimators=100, verbose=1, n_jobs=-1)
print("Training meta-model (Random Forest)...")
start_time_rf = time.time()
rf_meta.fit(stacked_oof, y_train)
print(f"Meta-model training completed in {time.time() - start_time_rf:.2f} seconds")

# Save Random Forest meta-model
print("Saving Random Forest meta-model...")
joblib.dump(rf_meta, 'rf_meta_model.pkl')

# --- Final Prediction and Evaluation ---
final_pred = rf_meta.predict(stacked_test)
accuracy = accuracy_score(y_test, final_pred)
print(f'Hybrid Model Accuracy: {accuracy:.4f}')

# --- LIME Explanation ---
# Define a prediction function for the stacked model
def predict_fn(X):
    lgb_pred = lgb_model.predict(X)
    svm_pred = svm_model.predict_proba(X)
    stacked_pred = np.hstack([lgb_pred, svm_pred])
    final_pred = rf_meta.predict_proba(stacked_pred)
    return final_pred

# Create a LIME explainer with placeholder feature names and disable discretization
print("Creating LIME explainer...")
class_names = ['Minor', 'Severe', 'Fatal']  # Adjust based on your target variable
feature_names = [f"feature_{i}" for i in range(X_train_ohe.shape[1])]  # Placeholder feature names
explainer = LimeTabularExplainer(
    training_data=X_train_ohe,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    discretize_continuous=False  # Disable discretization for binary features
)

# Explain a single prediction
print("Explaining a prediction with LIME...")
instance = X_test_ohe[0]  # First test instance
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=predict_fn,
    num_features=5  # Show the top 5 most influential features
)

# Print the explanation
print("LIME Explanation for a Single Instance:")
for feature, weight in exp.as_list():
    print(f"{feature}: {weight}")

# Save the explanation as a plot
exp.save_to_file('lime_explanation.html')

# Compute average feature importance across multiple instances
print("Computing average feature importance with LIME...")
num_instances = 100  # Number of instances to explain
feature_importance = np.zeros(X_train_ohe.shape[1])
for i in tqdm(range(num_instances), desc="LIME Explanations"):
    instance = X_test_ohe[i]
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_fn,
        num_features=5
    )
    for feature, weight in exp.as_list():
        feature_idx = feature_names.index(feature)  # This should now work
        feature_importance[feature_idx] += abs(weight)

# Normalize the importance scores
feature_importance /= num_instances
feature_importance /= feature_importance.sum()  # Normalize to sum to 1

# Print the top features
print("Average Feature Importance (Normalized):")
top_features = np.argsort(feature_importance)[::-1][:5]  # Top 5 features
for idx in top_features:
    print(f"{feature_names[idx]}: {feature_importance[idx]:.2f}")

print(f"Total training and explanation completed in {time.time() - start_time:.2f} seconds")