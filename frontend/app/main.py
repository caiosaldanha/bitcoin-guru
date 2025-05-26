from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL", "https://bitcoinguru.ml.caiosaldanha.com/api")

app = Flask(__name__)

@app.route("/")
def index():
    try:
        r = requests.get(f"{API_URL}/predict", timeout=10)
        r.raise_for_status()
        pred = r.json()
        # Adaptar para formato esperado pelo template (data orient='split' para dict de listas)
        if 'data' in pred and isinstance(pred['data'], dict) and 'columns' in pred['data'] and 'data' in pred['data']:
            # Converter orient='split' para dict de listas por coluna
            columns = pred['data']['columns']
            values = pred['data']['data']
            pred['data'] = {col: [row[i] for row in values] for i, col in enumerate(columns)}
    except Exception as e:
        pred = {'error': f'Erro ao acessar API /predict: {e}'}
    try:
        r_hist = requests.get(f"{API_URL}/history?limit=10", timeout=10)
        r_hist.raise_for_status()
        hist = r_hist.json()
        if 'data' in hist and isinstance(hist['data'], dict) and 'columns' in hist['data'] and 'data' in hist['data']:
            columns = hist['data']['columns']
            values = hist['data']['data']
            hist['data'] = {col: [row[i] for row in values] for i, col in enumerate(columns)}
    except Exception as e:
        hist = {'error': f'Erro ao acessar API /history: {e}'}
    return render_template("index.html", pred=pred, hist=hist)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
