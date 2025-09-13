# train_model.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]
TARGET = "target"

def load_or_generate():
    path = "heart.csv"
    if os.path.exists(path):
        print("Found heart.csv, loading dataset...")
        df = pd.read_csv(path)
        use_cols = [c for c in FEATURE_NAMES + [TARGET] if c in df.columns]
        df = df[use_cols].dropna()
        return df
    else:
        print("No heart.csv found - generating synthetic dataset (demo).")
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=1000,
            n_features=len(FEATURE_NAMES),
            n_informative=8,
            n_redundant=2,
            random_state=42
        )
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df[TARGET] = y
        df["age"] = (np.abs(df["age"]) * 7 + 50).astype(int)
        df["sex"] = (df["sex"] > 0).astype(int)
        df["cp"] = (np.abs(df["cp"]) % 4).astype(int)
        df["trestbps"] = (np.abs(df["trestbps"]) * 10 + 120).astype(int)
        df["chol"] = (np.abs[df["chol"]) * 10 + 200).astype(int)
        df["fbs"] = (df["fbs"] > 0).astype(int)
        df["restecg"] = (np.abs(df["restecg"]) % 3).astype(int)
        df["thalach"] = (np.abs(df["thalach"]) * 10 + 120).astype(int)
        df["exang"] = (df["exang"] > 0).astype(int)
        df["oldpeak"] = np.abs(df["oldpeak"]) * 3
        df["slope"] = (np.abs(df["slope"]) % 3).astype(int)
        df["ca"] = (np.abs(df["ca"]) % 4).astype(int)
        df["thal"] = ((np.abs(df["thal"]) % 3) + 1).astype(int)
        return df

def train_and_save():
    df = load_or_generate()
    if TARGET not in df.columns:
        raise RuntimeError("Dataset missing 'target' column.")

    X = df[FEATURE_NAMES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    trained_models = {}
    accuracies = {}
    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        preds = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = clf
        accuracies[name] = round(acc * 100, 2)
        print(f"{name} Accuracy: {accuracies[name]}%")

    artifact = {
        "scaler": scaler,
        "models": trained_models,
        "features": FEATURE_NAMES,
        "accuracies": accuracies
    }
    joblib.dump(artifact, "model.pkl")
    print("Saved model artifact to model.pkl")

if __name__ == "__main__":
    train_and_save()
