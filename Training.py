"""
Created on Sat Aug 30 19:09:49 2025

@author: Anirban Boral
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

dataset = pd.read_csv("Data/ProjectData3.csv")

print(f"Dataset shape: {dataset.shape}")
print(f"Columns: {list(dataset.columns)}")

dataset['Arrival_Date'] = pd.to_datetime(dataset['Arrival_Date'], errors='coerce')
dataset['Year'] = dataset['Arrival_Date'].dt.year
dataset['Month'] = dataset['Arrival_Date'].dt.month
dataset['Day'] = dataset['Arrival_Date'].dt.day

categorical_cols = ["State", "District", "Market", "Commodity", "Variety", "Grade"]
print(f"Categorical columns for encoding: {categorical_cols}")

# One-hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_features = encoder.fit_transform(dataset[categorical_cols])
print(f"\nEncoded categorical features shape: {encoded_features.shape}")

numeric_cols = ['Year', 'Month', 'Day', 'Commodity_Code', 'Min_Price', 'Max_Price']
numeric_features = dataset[numeric_cols].fillna(0).values
print(f"Numeric features shape: {numeric_features.shape}")

X = np.hstack([encoded_features, numeric_features])
print(f"Final feature matrix shape: {X.shape}")

# Target variable 
y = dataset["Modal_Price"]
print(f"Target variable shape: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Linear Regression Model
modelL = LinearRegression()
modelL.fit(X_train, y_train)

# Predictions
y_pred = modelL.predict(X_test)

mse1 = mean_squared_error(y_test, y_pred)
rmse1 = np.sqrt(mse1)
mae1 = mean_absolute_error(y_test, y_pred)
r21 = r2_score(y_test, y_pred)

print("\n=== Linear Regression Performance Metrics ===")
print(f"Mean Squared Error (MSE): {mse1:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse1:,.2f}")
print(f"Mean Absolute Error (MAE): {mae1:,.2f}")
print(f"R-squared (R²): {r21:.4f}")

# Prediction plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('Linear Regression: Actual vs Predicted Agricultural Prices', fontsize=14, fontweight='bold')

# Perfect prediction line
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
plt.savefig('Outputs/agricultural_price_predictionL.png', dpi=300, bbox_inches='tight')

# R2 score added to the plot
plt.text(0.05, 0.95, f'R² = {r21:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# XGBoost Model
modelX = XGBRegressor(
    n_estimators=100,           # No of trees
    max_depth=6,                # Maxi depth of trees
    learning_rate=0.1,          # Step size shrinkage
    subsample=0.8,              # Fraction of samples used for training each tree
    colsample_bytree=0.8,       # Fraction of features used for training each tree
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Use all available cores
)
modelX.fit(X_train,y_train)

# Make predictions
y_pred = modelX.predict(X_test)

# Calculate performance metrics  
mse2 = mean_squared_error(y_test, y_pred)
rmse2 = np.sqrt(mse2)
mae2 = mean_absolute_error(y_test, y_pred)
r22 = r2_score(y_test, y_pred)

print("\n=== XGBoost Performance Metrics ===")
print(f"Mean Squared Error (MSE): {mse2:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse2:,.2f}")
print(f"Mean Absolute Error (MAE): {mae2:,.2f}")
print(f"R-squared (R²): {r22:.4f}")

# Create and save the prediction plot
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
plt.xlabel('Actual Price', fontsize=12)
plt.ylabel('Predicted Price', fontsize=12)
plt.title('XGBoost: Actual vs Predicted Agricultural Prices', fontsize=14, fontweight='bold')

plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction Line')
plt.savefig('Outputs/agricultural_price_predictionX.png', dpi=300, bbox_inches='tight')

# Add R² score to the plot
plt.text(0.05, 0.95, f'R² = {r22:.4f}', transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

model_data1 = {
    'model': modelX,
    'encoder': encoder,
    'categorical_cols': ["State", "District", "Market", "Commodity", "Variety", "Grade"],
    'performance': {'r2': r22, 'rmse': rmse2},
    'numeric_cols': ['Year', 'Month', 'Day', 'Commodity_Code', 'Min_Price', 'Max_Price']
}
model_data2 = {
    'model': modelL,
    'encoder': encoder,
    'categorical_cols': ["State", "District", "Market", "Commodity", "Variety", "Grade"],
    'performance': {'r2': r21, 'rmse': rmse1},
    'numeric_cols': ['Year', 'Month', 'Day', 'Commodity_Code', 'Min_Price', 'Max_Price']
}
# Save the models
joblib.dump(model_data2, "Models/agri_modelL.joblib")
joblib.dump(model_data1, "Models/agri_modelX.joblib") 
#For big model size, we can add another argument "compress=3" to lower the size, but that makes the loading slower


