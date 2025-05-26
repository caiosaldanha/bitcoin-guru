from flask import Flask, render_template, jsonify, request
import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL", "https://bitcoinguru.ml.caiosaldanha.com/api")

app = Flask(__name__)

def convert_to_col_dict(data):
    # Se já for dict de listas, retorna direto
    if isinstance(data, dict) and 'date' in data:
        return data
    # Se vier como orient='split'
    if isinstance(data, dict) and 'columns' in data and 'data' in data:
        columns = data['columns']
        values = data['data']
        return {col: [row[i] for row in values] for i, col in enumerate(columns)}
    # Se vier como lista de dicts (orient='records')
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        out = {k: [] for k in data[0].keys()}
        for row in data:
            for k, v in row.items():
                out[k].append(v)
        return out
    return {}

@app.route("/")
def index():
    try:
        r = requests.get(f"{API_URL}/predict", timeout=10)
        r.raise_for_status()
        pred = r.json()
        print("[DEBUG] /predict raw:", pred)
        if 'data' in pred:
            pred['data'] = convert_to_col_dict(pred['data'])
            print("[DEBUG] /predict converted:", pred['data'])
    except Exception as e:
        pred = {'error': f'Erro ao acessar API /predict: {e}'}
    try:
        r_hist = requests.get(f"{API_URL}/history?limit=10", timeout=10)
        r_hist.raise_for_status()
        hist = r_hist.json()
        print("[DEBUG] /history raw:", hist)
        if 'data' in hist:
            hist['data'] = convert_to_col_dict(hist['data'])
            print("[DEBUG] /history converted:", hist['data'])
    except Exception as e:
        hist = {'error': f'Erro ao acessar API /history: {e}'}
    return render_template("index.html", pred=pred, hist=hist)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
