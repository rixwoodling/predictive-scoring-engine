import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV
def load_data():
    return pd.read_csv("csv/loan_data.csv")

# Detect binary columns and prompt user to select target
def select_target_column(df):
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]

    if not binary_cols:
        raise ValueError("No binary columns found in dataset.")

    print("\nDetected binary columns:")
    for i, col in enumerate(binary_cols):
        print(f"{i}: {col}")

    while True:
        try:
            choice = int(input("\nSelect target column index: "))
            return binary_cols[choice]
        except (ValueError, IndexError):
            print("Invalid selection. Try again.")

# Split features and target
def split_features_target(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Convert target to numeric if needed
    if y.dtype == "object":
        y = y.astype("category").cat.codes

    return X, y

# Identify numeric and categorical columns
def get_column_types(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns
    return num_cols, cat_cols

# Build preprocessing + model pipeline
def build_pipeline(num_cols, cat_cols):
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

# Train model
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

# Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")

# Save trained model
def save_model(model):
    joblib.dump(model, "model.pkl")
    print("Model saved as model.pkl")

def main():
    print("Predictive Scoring Engine (ML Pipeline)")
    df = load_data()
    print(df.head())
    target_col = select_target_column(df)
    X, y = split_features_target(df, target_col)
    num_cols, cat_cols = get_column_types(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline = build_pipeline(num_cols, cat_cols)
    model = train_model(pipeline, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)
    
if __name__ == "__main__":
    main()


