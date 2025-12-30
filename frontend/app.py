"""
Flask frontend for Quantum Computing LLM.
Serves the chat UI and proxies API requests to the backend.

Local:
    cd frontend
    python app.py

Railway:
    Set BACKEND_URL environment variable to backend service URL
"""

import os
from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend API URL (env var for Railway, localhost for local dev)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


@app.route("/")
def index():
    """Serve the main chat page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Proxy query requests to the backend API."""
    try:
        data = request.get_json()
        response = requests.post(
            f"{BACKEND_URL}/query",
            json=data,
            timeout=120
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to backend. Is it running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def api_health():
    """Proxy health check to the backend API."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e), "backend": "unavailable"}), 503


@app.route("/health", methods=["GET"])
def health():
    """Frontend health check for Railway."""
    return jsonify({"status": "ok", "service": "frontend"}), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 3000))
    app.run(host="0.0.0.0", port=port, debug=True)
