"""
Flask frontend for Quantum Computing LLM.
Serves the chat UI and proxies API requests to the backend.

Run:
    cd frontend
    python app.py
"""

from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

# Backend API URL
BACKEND_URL = "http://localhost:8000"


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
            timeout=120  # 2 min timeout for slow responses
        )
        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out. Please try again."}), 504
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot connect to backend. Is it running?"}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """Proxy health check to the backend API."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e), "backend": "unavailable"}), 503


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
