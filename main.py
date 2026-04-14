import sys
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

def load_data():
    datasets = ["titanic", "penguins", "tips", "iris"]
    name = random.choice(datasets)
    
    df = sns.load_dataset(name)
    print(f"\nLoaded dataset: {name}")
    return df

# Detect binary columns and prompt user to select target
def select_target_column(df):
    binary_cols = [
        col for col in df.columns
        if df[col].nunique() == 2
    ]
    if not binary_cols:
        print("\nNo binary columns found in dataset.")
        print("This pipeline currently expects a binary classification target.")
        sys.exit(1)

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
    overview = df.dtypes.to_frame(name="Dtype")
    overview["Non-Null"] = df.count()
    overview["Missing"] = df.isnull().sum()
    overview["Missing %"] = (overview["Missing"] / len(df) * 100).round(2)

    print("\nDATA OVERVIEW")
    print(overview)

    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # ---- DESCRIPTIVE STATS ----
    print("\nDESCRIBE")
    print(df.describe().round(3))

    # ---- TARGET DISTRIBUTION (ASCII) ----
    counts = df[target_col].value_counts().sort_index()
    data = [(str(label), int(count)) for label, count in counts.items()]

    print("\nTARGET DISTRIBUTION")
    max_count = max(v for _, v in data)

    for label, count in data:
        bar_len = int((count / max_count) * 20)
        bar = "*" * bar_len
        print(f"{label:<6} | {count:<5} {bar}")

    # ---- TARGET DISTRIBUTION PLOT ----
    sns.countplot(x=target_col, data=df)
    plt.title(f"{target_col} Distribution")
    plt.savefig("target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

# Split features and target
def split_features_target(df, target_col):
    df = df[df[target_col].notna()]
    
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

def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True)
    }
def evaluate_pipeline(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    # Get probability scores or decision function
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)
        # handle binary vs multi-class
        if y_proba.shape[1] == 2:
            y_proba = y_proba[:, 1]
    else:
        y_proba = pipeline.decision_function(X_test)
    # ROC-AUC handling (binary vs multi-class)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc = roc_auc_score(y_test, y_proba, multi_class="ovr")

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1": f1_score(y_test, y_pred, average="weighted"),
        "ROC-AUC": roc
    }

def choose_model(num_cols, cat_cols, X_train, X_test, y_train, y_test):
    models = get_models()

    results = []
    best_model = None
    best_score = -1
    best_name = ""

    print("\nMODEL COMPARISON")

    for name, model in models.items():
        pipeline = build_pipeline(num_cols, cat_cols, model)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_pipeline(pipeline, X_test, y_test)

        print(f"{name} → Accuracy: {metrics['Accuracy']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

        results.append({"Model": name, **metrics})

        if metrics["ROC-AUC"] > best_score:
            best_score = metrics["ROC-AUC"]
            best_model = model
            best_name = name

    df_results = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)

    print("\nFinal Results:")
    print(df_results.round(3))

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
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    best_pipeline = choose_model(
        num_cols, cat_cols, X_train, X_test, y_train, y_test
    )
    
    evaluate_model(best_pipeline, X_test, y_test)
    save_model(best_pipeline)
    
if __name__ == "__main__":
    main()




