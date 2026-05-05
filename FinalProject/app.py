from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model and feature names ──────────────────────────────────────────────
model_path    = os.path.join(os.path.dirname(__file__), "xgb_loan_model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "feature_names.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(features_path, "rb") as f:
    feature_names = pickle.load(f)

print("✅ Model loaded")
print("✅ Features:", feature_names)

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Build feature array in exact same order as training
        features = []
        for col in feature_names:
            val = data.get(col, 0)
            features.append(float(val))

        features_array = np.array([features])

        # Predict
        prediction  = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]

        approved_prob = round(float(probability[1]) * 100, 1)
        rejected_prob = round(float(probability[0]) * 100, 1)

        result = {
            "prediction"   : "Approved" if prediction == 1 else "Rejected",
            "approved_prob": approved_prob,
            "rejected_prob": rejected_prob,
            "status"       : "success"
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)