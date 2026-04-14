import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import asciibars
import numpy as np

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
    
    # Compute target distribution, convert to asciibars format, print:
    counts = df[target_col].value_counts().sort_index()
    data = [(str(label), int(count)) for label, count in counts.items()]
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
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    return num_cols, cat_cols

def build_pipeline(num_cols, cat_cols, model):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

def verify_pipeline(num_cols, cat_cols, X):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])
    X_transformed = preprocessor.fit_transform(X)

    print("\nPIPELINE VERIFICATION")
    print("Original:", X.shape)
    print("Transformed:", X_transformed.shape)
    print("Missing after pipeline?", np.isnan(X_transformed).any())

def pick_best_model(models, num_cols, cat_cols, X_train, X_test, y_train, y_test):
    best_model = None
    best_score = -1
    best_name = ""

    print("\nMODEL COMPARISON")

    for name, model in models.items():
        pipeline = build_pipeline(num_cols, cat_cols, model)
        pipeline.fit(X_train, y_train)
        acc = pipeline.score(X_test, y_test)

        print(f"{name} → Accuracy: {acc:.4f}")

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name
            
    print(f"\nBest model: {best_name}")
    return best_model

# Train model
def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

# Evaluate model
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.3f}")

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
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Optional: verify preprocessing (training data only)
    verify_pipeline(num_cols, cat_cols, X_train)

    # 8. Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True)
    }

    # 9. Pick best model
    best_model = pick_best_model(
        models, num_cols, cat_cols,
        X_train, X_test, y_train, y_test
    )

    # 10. Build final pipeline
    pipeline = build_pipeline(num_cols, cat_cols, best_model)

    # 11. Train final model
    pipeline = train_model(pipeline, X_train, y_train)

    # 12. Evaluate final model
    evaluate_model(pipeline, X_test, y_test)

    # 13. Save model
    save_model(pipeline)
    
if __name__ == "__main__":
    main()


