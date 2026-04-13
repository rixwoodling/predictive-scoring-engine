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
    return pd.read_csv("data.csv")


# Build preprocessing + model pipeline
def build_pipeline(num_cols, cat_cols):

    # Apply scaling to numeric and encoding to categorical features
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    # Combine preprocessing and model into one pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier())
    ])

    return pipeline


def main():

    print("Predictive Scoring Engine (ML Pipeline)")

    # Load data into DataFrame
    df = load_data()

    # Split features and target column
    X = df.drop("default", axis=1)
    y = df["default"]

    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize pipeline
    pipeline = build_pipeline(num_cols, cat_cols)

    # Train model on training data
    pipeline.fit(X_train, y_train)

    # Generate predictions on test data
    preds = pipeline.predict(X_test)

    # Compute accuracy metric
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.3f}")

    # Save trained pipeline to disk
    joblib.dump(pipeline, "model.pkl")
    print("Model saved as model.pkl")


if __name__ == "__main__":
    main()
