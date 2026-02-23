import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error
import pickle
import os

# Define file paths
DATA_PATH = "engine_data.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

def main():
    print("Loading data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Simple data cleaning checks
    print(f"Dataset shape: {df.shape}")
    
    # Handle missing values if any
    df = df.dropna()

    # Define features and target
    target_col = 'Engine Condition'
    feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y_class = df[target_col]

    print(f"\nFeature columns: {feature_cols}")
    print(f"Target distribution:\n{y_class.value_counts()}")

    # Split data (ensure everything is aligned)
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )

    # Scale data
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Classifiers for Engine Health ---
    print("\nTraining Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train_class)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test_class, rf_preds)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")

    print("Training Gradient Boosting Classifier...")
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_model.fit(X_train_scaled, y_train_class)
    gb_preds = gb_model.predict(X_test_scaled)
    gb_acc = accuracy_score(y_test_class, gb_preds)
    print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

    # Select best model
    best_model = None
    best_name = ""
    if rf_acc >= gb_acc:
        best_model = rf_model
        best_name = "Random Forest"
    else:
        best_model = gb_model
        best_name = "Gradient Boosting"

    print(f"\nSelected {best_name} as the best classifer model.")
    
    # Save models and scaler
    print(f"\nSaving {best_name} model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
        
    # Also save feature names into scaler if not present, but typically we can reconstruct them.
    # We will just write the scaler.
    print(f"Saving scaler to {SCALER_PATH}...")
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    print("Training complete!")

if __name__ == "__main__":
    main()
