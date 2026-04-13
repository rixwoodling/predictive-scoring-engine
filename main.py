import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import asciibars

from sklearn.impute import SimpleImputer
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
            
# Run basic EDA and save target distribution plot
def run_eda(df, target_col):
    # Print structure, stats, and missing values
    print("\nDATA INFO")
    df.info()

    print("\nDESCRIBE")
    print(df.describe().round(3))

    print("\nMISSING VALUES")
    print(df.isnull().sum())
    
    # Compute target distribution
    counts = df[target_col].value_counts().sort_index()
    # Convert to asciibars format: list of (label, value)
    data = [(str(label), int(count)) for label, count in counts.items()]
    # CLI bar chart
    print("\nTARGET DISTRIBUTION")
    max_count = max(v for _, v in data)

    for label, count in data:
        bar_len = int((count / max_count) * 20)
        bar = "*" * bar_len
        print(f"{label:<6} | {count:<5} {bar}")
        
    # Save matplotlib plot (non-blocking)
    sns.countplot(x=target_col, data=df)
    plt.title(f"{target_col} Distribution")
    plt.savefig("target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Print ASCII bar chart with stars
def print_ascii_bar(y):
    counts = y.value_counts().sort_index()
    max_count = counts.max()

    for label, count in counts.items():
        bar_len = int((count / max_count) * 20)
        bar = "*" * bar_len
        print(f"{str(label):<6} | {count:<5} {bar}")

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
    target_col = select_target_column(df)
    run_eda(df, target_col)

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


