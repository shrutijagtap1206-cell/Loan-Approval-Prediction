"""
app.py
======
Flask Web Application & API layer for the Loan Application Approval Prediction System.

Endpoints:
  - GET  /             -> Serves modern frontend interface (static/index.html)
  - POST /predict      -> Core prediction API receiving applicant details
  - POST /api/predict  -> Alias for /predict
  - GET  /api/health   -> Health check and model metadata
"""

import os
from flask import Flask, request, jsonify, send_from_directory
from engine import predict, get_model

app = Flask(__name__, static_folder="static")


@app.route("/")
def home():
    """Serves the loan application frontend."""
    return send_from_directory("static", "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Returns application health and model bundle status."""
    try:
        bundle = get_model()
        return jsonify({
            "status": "healthy",
            "model_name": bundle.get("model_name", "Unknown"),
            "features_count": len(bundle.get("feature_order", [])),
            "metrics": bundle.get("metrics", {})
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Main prediction endpoint.
    Accepts applicant JSON payload matching Section 23 specification.
    """
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({
            "success": False,
            "errors": ["Invalid JSON request payload."]
        }), 400

    try:
        result = predict(payload)
        status_code = 200 if result.get("success", False) else 400
        return jsonify(result), status_code
    except Exception as e:
        return jsonify({
            "success": False,
            "errors": [f"Server error during prediction: {str(e)}"]
        }), 500


if __name__ == "__main__":
    # Ensure model is loadable on startup
    try:
        b = get_model()
        print(f"Loaded model: {b.get('model_name')}")
    except Exception as ex:
        print(f"Warning: Could not preload model: {ex}")

    port = int(os.environ.get("PORT", 5000))
    print(f"Starting Loan Approval System on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
