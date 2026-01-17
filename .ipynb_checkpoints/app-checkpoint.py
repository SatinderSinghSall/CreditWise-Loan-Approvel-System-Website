from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("loan_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        required_fields = [
            "Applicant_Income",
            "Coapplicant_Income",
            "Age",
            "Credit_Score",
            "DTI_Ratio",
            "Loan_Amount",
            "Loan_Term",
            "Savings",
            "Collateral_Value",
            "Existing_Loans"
        ]

        # Check for missing fields
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[data[field] for field in required_fields]])

        features = scaler.transform(features)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]

        return jsonify({
            "approved": int(prediction),
            "probability": round(float(probability), 2)
        })

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
