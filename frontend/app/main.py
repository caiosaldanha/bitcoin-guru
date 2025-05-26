from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd

API_URL = "http://backend:8000/api"

app = Flask(__name__)

@app.route("/")
def index():
    # Última previsão
    r = requests.get(f"{API_URL}/predict")
    pred = r.json()
    # Histórico
    r_hist = requests.get(f"{API_URL}/history?limit=10")
    hist = r_hist.json()
    return render_template("index.html", pred=pred, hist=hist)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
