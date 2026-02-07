from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os
import warnings
import requests
from flask_cors import CORS
from utils.shap_explainer import explain_prediction

warnings.filterwarnings("ignore")

# ============================
# CONFIG
# ============================

MODEL_ID = "1PzmC8ZDjyFP6-Ewp7SWJgngFfIWquj-D"

MODEL_URL = f"https://drive.google.com/uc?export=download&id={MODEL_ID}"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")


# ============================
# FLASK APP
# ============================

app = Flask(__name__)

CORS(app, resources={r"/predict": {"origins": "*"}})


# ============================
# DOWNLOAD MODEL
# ============================

def download_model():
    """
    Download ML model from Google Drive if not present
    """

    if os.path.exists(MODEL_PATH):
        print("✅ Model already exists")
        return

    print("⬇️ Downloading model from Google Drive...")

    response = requests.get(MODEL_URL, stream=True)

    if response.status_code != 200:
        raise Exception("❌ Failed to download model")

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Model downloaded successfully")


# ============================
# LOAD MODEL
# ============================

download_model()

model = joblib.load(MODEL_PATH)

print("🚀 Model loaded successfully")


# ============================
# HELPERS
# ============================

def _coerce_mostly_numeric(df: pd.DataFrame, threshold: float = 0.8):

    obj_cols = df.select_dtypes(include=["object"]).columns

    for c in obj_cols:

        coerced = pd.to_numeric(df[c], errors="coerce")

        if coerced.notna().mean() >= threshold:
            df[c] = coerced

    return df


def _align_to_model_features(df: pd.DataFrame):

    if hasattr(model, "feature_names_in_"):

        expected = list(model.feature_names_in_)

        present = set(df.columns)

        missing = [c for c in expected if c not in present]

        for c in missing:
            df[c] = pd.NA

        df = df.reindex(columns=expected)

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        cat_cols = [c for c in df.columns if c not in num_cols]

        if num_cols:
            df[num_cols] = df[num_cols].fillna(0)

        for c in cat_cols:
            df[c] = df[c].astype("string").fillna("unknown")

    return df


# ============================
# ROUTES
# ============================

@app.route("/")
def home():
    return "🚀 Fraud Detection API is Running"


@app.route("/predict", methods=["POST"])
def predict():

    # ------------------------
    # File Check
    # ------------------------

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400


    # ------------------------
    # Read File
    # ------------------------

    try:

        fname = file.filename.lower()

        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)

        elif fname.endswith(".csv"):
            df = pd.read_csv(file, encoding="utf-8", low_memory=False)

        else:
            return jsonify({"error": "Upload CSV or Excel"}), 400

    except Exception as e:
        return jsonify({"error": f"Read error: {e}"}), 400


    print("📂 File shape:", df.shape)


    # ------------------------
    # Clean Data
    # ------------------------

    df.columns = [c.strip() for c in df.columns]


    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].replace(
            {"": np.nan, "nan": np.nan, "None": np.nan, "NA": np.nan}
        )


    df = _coerce_mostly_numeric(df)


    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    cat_cols = [c for c in df.columns if c not in num_cols]


    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


    for c in cat_cols:

        s = df[c].astype("string").str.strip()

        s = s.replace({"": pd.NA})

        df[c] = s.fillna("unknown")


    if num_cols:
        df[num_cols] = df[num_cols].fillna(0)


    df = _align_to_model_features(df)


    print("🔍 Types:\n", df.dtypes)


    # ------------------------
    # Predict
    # ------------------------

    try:

        preds = model.predict(df)

    except Exception as e:

        print("❌ Prediction Error:", e)

        return jsonify({"error": f"Prediction failed: {e}"}), 500


    # ------------------------
    # Build Response
    # ------------------------

    results = []

    for i, p in enumerate(preds):

        label = "Fraudulent" if int(p) == 1 else "Legitimate"


        first = str(df.iloc[i]["first"]) if "first" in df.columns else "N/A"

        last = str(df.iloc[i]["last"]) if "last" in df.columns else "N/A"


        try:
            reason = explain_prediction(model, df.iloc[[i]])

        except:
            reason = "Explanation not available"


        results.append({
            "index": i + 1,
            "first": first,
            "last": last,
            "label": label,
            "reason": reason
        })


    return jsonify(results)


# ============================
# MAIN
# ============================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port, debug=True)
