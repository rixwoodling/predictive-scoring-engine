import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    print("Predictive Scoring Engine (ML Pipeline)")

    data = pd.DataFrame({
        "income": [50000, 60000, 40000, 80000],
        "credit_score": [600, 650, 580, 720],
        "default": [1, 0, 1, 0]
    })

    X = data[["income", "credit_score"]]
    y = data["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Combine preprocessing (scaling) and model into a single reusable pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression())
    ])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print("\nAccuracy:", acc)
    print("Predictions:", preds)

if __name__ == "__main__":
    main()
