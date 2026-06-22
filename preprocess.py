# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
import time

# Load the dataset
print("Loading dataset...")
start_time = time.time()
df = pd.read_csv('cleaned_2.csv')
print(f"Dataset loaded in {time.time() - start_time:.2f} seconds")

# Handle missing values
print("Handling missing values...")
df.fillna(df.mode().iloc[0], inplace=True)

# Identify categorical features
cat_features = [col for col in df.columns if df[col].dtype == 'object' and col != 'Accident_severity']

# Encode categorical features
print("Encoding categorical features...")
X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']

# Label encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Label encode categorical features for LightGBM
X_encoded = X.copy()
label_encoders = {}
for col in cat_features:
    le_col = LabelEncoder()
    X_encoded[col] = le_col.fit_transform(X[col])
    label_encoders[col] = le_col

# One-hot encode for SVM and Random Forest
print("Performing one-hot encoding...")
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_ohe = ohe.fit_transform(X_encoded)

# Balance the dataset using SMOTE
print("Balancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_encoded, y_encoded)
X_ohe_balanced, y_ohe_balanced = smote.fit_resample(X_ohe, y_encoded)

# Convert balanced data back to DataFrame for LightGBM (preserving column names)
X_balanced_df = pd.DataFrame(X_balanced, columns=X_encoded.columns)
y_balanced_df = pd.Series(y_balanced, name='Accident_severity')

# Save balanced datasets
print("Saving balanced datasets...")
X_balanced_df.to_csv('balanced_X.csv', index=False)
y_balanced_df.to_csv('balanced_y.csv', index=False)
np.save('balanced_X_ohe.npy', X_ohe_balanced)  # Save as numpy array for efficiency
np.save('balanced_y_ohe.npy', y_ohe_balanced)
print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

# Print class distribution before and after SMOTE
print("\nOriginal class distribution:")
print(pd.Series(y_encoded).value_counts())
print("\nBalanced class distribution:")
print(pd.Series(y_balanced).value_counts())