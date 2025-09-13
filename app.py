# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load saved artifact
artifact = joblib.load("model.pkl")
models = artifact["models"]
scaler = artifact["scaler"]
FEATURE_NAMES = artifact["features"]
accuracies = artifact["accuracies"]

# Mappings for categorical fields
MAPPINGS = {
    "sex": {0: "Female", 1: "Male"},
    "fbs": {0: "No", 1: "Yes"},
    "exang": {0: "No", 1: "Yes"},
    "cp": {
        0: "Asymptomatic",
        1: "Atypical Angina",
        2: "Non-Anginal Pain",
        3: "Typical Angina",
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "LV Hypertrophy",
    },
    "slope": {
        0: "Downsloping",
        1: "Flat",
        2: "Upsloping",
    },
    "thal": {
        1: "Normal",
        2: "Fixed Defect",
        3: "Reversible Defect",
    }
}

# Friendly display names
DISPLAY_NAMES = {
    "age": "Age",
    "sex": "Gender",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Slope",
    "ca": "Number of Major Vessels",
    "thal": "Thalassemia"
}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    inputs = {}
    try:
        for f in FEATURE_NAMES:
            val = request.form.get(f)
            if val is None:
                return f"Missing value for {f}", 400
            if f == "oldpeak":
                inputs[f] = float(val)
            else:
                inputs[f] = int(float(val))
    except Exception as e:
        return f"Invalid input: {e}", 400

    X = pd.DataFrame([inputs], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)

    details = {}
    chart_labels = []
    chart_values = []

    for name, clf in models.items():
        prob = clf.predict_proba(X_scaled)[0][1]
        details[name] = f"{prob*100:.1f}%"
        chart_labels.append(name)
        chart_values.append(round(prob*100, 1))

    avg_prob = sum(chart_values) / len(chart_values)
    percent = round(avg_prob, 1)

    acc_labels = list(accuracies.keys())
    acc_values = list(accuracies.values())

    # Convert to readable format
    readable_inputs = {}
    for k, v in inputs.items():
        label = DISPLAY_NAMES.get(k, k)
        if k in MAPPINGS:
            readable_inputs[label] = MAPPINGS[k].get(v, v)
        else:
            readable_inputs[label] = v

    return render_template(
        "result.html",
        inputs=readable_inputs,
        percent=percent,
        details=details,
        chart_labels=chart_labels,
        chart_values=chart_values,
        acc_labels=acc_labels,
        acc_values=acc_values
    )

if __name__ == "__main__":
    app.run(debug=True)
