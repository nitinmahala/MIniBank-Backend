from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os
import warnings
from utils.shap_explainer import explain_prediction
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# ✅ Load your trained model
model_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
model = joblib.load(model_path)
print("✅ Model loaded successfully")

@app.route('/')
def index():
    return render_template('index.html')


def _coerce_mostly_numeric(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Try to convert object columns to numeric if mostly numeric-like."""
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= threshold:
            df[c] = coerced
    return df


def _align_to_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Align input DataFrame with model's expected features."""
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        present = set(df.columns)
        to_add = [c for c in expected if c not in present]

        for c in to_add:
            df[c] = pd.NA

        df = df.reindex(columns=expected)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if num_cols:
            df[num_cols] = df[num_cols].fillna(0)

        for c in cat_cols:
            df[c] = df[c].astype("string").fillna("unknown")

    return df


@app.route('/predict', methods=['POST'])
def predict():
    # 1️⃣ File check
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename!"}), 400

    # 2️⃣ Read Excel or CSV
    try:
        fname = file.filename.lower()
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        elif fname.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", low_memory=False)
        else:
            return jsonify({"error": "Unsupported file type! Upload .csv or .xlsx"}), 400
    except Exception as e:
        return jsonify({"error": f"Error reading file: {e}"}), 400

    print("📂 Input shape:", df.shape)

    # 3️⃣ Clean & enforce consistent types
    df.columns = [c.strip() for c in df.columns]

    # Replace obvious empty strings with NaN
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].replace({"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan})

    # Try to coerce mostly numeric object columns
    df = _coerce_mostly_numeric(df, threshold=0.8)

    # Identify numeric vs categorical
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Convert numerics safely
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Convert categoricals safely
    for c in cat_cols:
        s = df[c].astype("string").str.strip()
        s = s.replace({"": pd.NA})
        df[c] = s.fillna("unknown")

    # Fill NaN for numerics
    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)

    # Optional: align to model features
    df = _align_to_model_features(df)

    print("🔍 Data types before prediction:\n", df.dtypes)

    # 4️⃣ Predict
    try:
        predictions = model.predict(df)
    except Exception as e:
        print("❌ Prediction failed:", e)
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # 5️⃣ Generate results with `first` and `last` fields
    results = []
    for i, pred in enumerate(predictions):
        label = "Fraudulent" if int(pred) == 1 else "Legitimate"

        # ✅ Extract first and last name
        first_name = str(df.iloc[i]["first"]) if "first" in df.columns else "N/A"
        last_name = str(df.iloc[i]["last"]) if "last" in df.columns else "N/A"

        # Get SHAP explanation (optional)
        try:
            reason = explain_prediction(model, df.iloc[[i]])
        except Exception:
            reason = "Explanation not available"

        results.append({
            "index": int(i + 1),
            "first": first_name,
            "last": last_name,
            "label": label,
            "reason": reason
        })

    return jsonify(results)


if __name__ == '__main__':
    # You can set host='0.0.0.0' for containerized environments
    app.run(debug=True)
