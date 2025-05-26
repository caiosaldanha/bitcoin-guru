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
        app.logger.info("[DEBUG] convert_to_col_dict columns: %s", columns)
        app.logger.info("[DEBUG] convert_to_col_dict values: %s", values)
        if not columns:
            app.logger.error("[DEBUG] columns vazio!")
            return {}
        if not values:
            app.logger.warning("[DEBUG] values vazio!")
            return {col: [] for col in columns}
        try:
            # Garante que values é lista de listas
            if isinstance(values[0], list):
                transposed = list(zip(*values))
            else:
                # Caso values seja uma lista simples (uma linha só, não aninhada)
                transposed = [[v] for v in values]
            app.logger.info("[DEBUG] transposed: %s", transposed)
            result = {col: list(transposed[i]) for i, col in enumerate(columns)}
            app.logger.info("[DEBUG] convert_to_col_dict result: %s", result)
            return result
        except Exception as e:
            app.logger.error("[DEBUG] Exception in transpose: %s", e)
            return {col: [] for col in columns}
    # Se vier como lista de dicts (orient='records')
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        out = {k: [] for k in data[0].keys()}
        for row in data:
            for k, v in row.items():
                out[k].append(v)
        app.logger.info("[DEBUG] convert_to_col_dict orient=records result: %s", out)
        return out
    app.logger.warning("[DEBUG] convert_to_col_dict: formato não reconhecido, retornando {}. Data: %s", data)
    return {}

@app.route("/")
def index():
    try:
        r = requests.get(f"{API_URL}/predict", timeout=10)
        r.raise_for_status()
        pred = r.json()
        app.logger.info("[DEBUG] /predict raw: %s", pred)
        if 'data' in pred:
            pred['data'] = convert_to_col_dict(pred)
            app.logger.info("[DEBUG] /predict converted: %s", pred['data'])
    except Exception as e:
        pred = {'error': f'Erro ao acessar API /predict: {e}'}
    
    try:
        r_hist = requests.get(f"{API_URL}/history?limit=10", timeout=10)
        r_hist.raise_for_status()
        hist = r_hist.json()
        app.logger.info("[DEBUG] /history raw: %s", hist)
        if 'data' in hist:
            hist['data'] = convert_to_col_dict(hist)
            app.logger.info("[DEBUG] /history converted: %s", hist['data'])
    except Exception as e:
        hist = {'error': f'Erro ao acessar API /history: {e}'}
    
    # Buscando dados de preços históricos
    try:
        r_prices = requests.get(f"{API_URL}/prices?days=30", timeout=10)
        r_prices.raise_for_status()
        prices = r_prices.json()
        app.logger.info("[DEBUG] /prices raw: %s", prices)
        if 'data' in prices:
            prices['data'] = convert_to_col_dict(prices)
            app.logger.info("[DEBUG] /prices converted: %s", prices['data'])
    except Exception as e:
        prices = {'error': f'Erro ao acessar API /prices: {e}'}
    
    return render_template("index.html", pred=pred, hist=hist, prices=prices)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
