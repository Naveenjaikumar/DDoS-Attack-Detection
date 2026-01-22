from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==============================
# Load Model & Scaler
# ==============================
model = load_model("ddos_model.h5")
scaler = joblib.load("scaler.pkl")

# ==============================
# Feature Order (MUST MATCH TRAINING)
# ==============================
FEATURES = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Total Length of Fwd Packets',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Average Packet Size',
    'Avg Fwd Segment Size',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Std'
]

# ==============================
# Prediction API
# ==============================
@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.json

        # Auto-map input features
        input_values = []
        for feature in FEATURES:
            if feature not in data:
                return jsonify({"error": f"Missing feature: {feature}"}), 400
            input_values.append(float(data[feature]))

        X = np.array(input_values).reshape(1, -1)
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0][0]

        result = "DDoS ATTACK ⚠️" if prediction > 0.5 else "Normal ✅"

        return jsonify({
            "prediction": result,
            "confidence": round(float(prediction), 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
