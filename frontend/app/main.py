from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
import requests
import pandas as pd
import os

API_URL = os.environ.get("API_URL", "https://bitcoinguru.ml.caiosaldanha.com/api")

app = Flask(__name__)

# Filtro personalizado para formatar números sem depender de locale
@app.template_filter('format_number')
def format_number(value):
    try:
        # Tenta converter para float primeiro (mais flexível)
        num = float(value)
        # Formatamos sempre com 2 casas decimais para valores monetários
        return "{:,.2f}".format(num).replace(",", ".")
    except (ValueError, TypeError):
        return value

def convert_to_col_dict(data):
    app.logger.warning(f"[DEBUG] convert_to_col_dict: tipo={type(data)}, chaves={list(data.keys()) if isinstance(data, dict) else 'N/A'}")
    # Se já for dict de listas, retorna direto
    if isinstance(data, dict) and isinstance(data.get('date'), list):
        app.logger.info("[DEBUG] Dados já estão no formato correto (dict de listas)")
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
        # Transpõe a matriz de dados para dict de listas
        transposed = list(zip(*values))
        result = {col: list(transposed[i]) for i, col in enumerate(columns)}
        # date_display fallback
        if 'date_display' not in result and 'date' in result:
            result['date_display'] = result['date']
        app.logger.info("[DEBUG] convert_to_col_dict result: %s", result)
        return result
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
    # Tentar obter a previsão mais recente
    try:
        app.logger.info("Buscando previsão em %s", f"{API_URL}/predict")
        r = requests.get(f"{API_URL}/predict", timeout=10)
        r.raise_for_status()
        pred = r.json()
        app.logger.info("[DEBUG] /predict raw: %s", pred)
        if 'data' in pred:
            pred = convert_to_col_dict(pred)
            app.logger.info("[DEBUG] /predict converted: %s", pred)
    except requests.exceptions.RequestException as e:
        app.logger.error("Erro ao acessar API /predict: %s", e)
        pred = {'error': f'Erro ao acessar API /predict: {e}'}
    except Exception as e:
        app.logger.error("Erro ao processar dados de /predict: %s", e)
        pred = {'error': f'Erro ao processar dados: {e}'}
    
    # Tentar obter o histórico de previsões
    try:
        app.logger.info("Buscando histórico em %s", f"{API_URL}/history?limit=10")
        r_hist = requests.get(f"{API_URL}/history?limit=10", timeout=10)
        r_hist.raise_for_status()
        hist = r_hist.json()
        app.logger.info("[DEBUG] /history raw: %s", hist)
        if 'data' in hist:
            hist = convert_to_col_dict(hist)
            app.logger.info("[DEBUG] /history converted: %s", hist)
    except requests.exceptions.RequestException as e:
        app.logger.error("Erro ao acessar API /history: %s", e)
        hist = {'error': f'Erro ao acessar API /history: {e}'}
    except Exception as e:
        app.logger.error("Erro ao processar dados de /history: %s", e)
        hist = {'error': f'Erro ao processar dados: {e}'}
    
    # Tentar obter preços históricos para o gráfico
    try:
        app.logger.info("Buscando preços em %s", f"{API_URL}/prices?days=30")
        r_prices = requests.get(f"{API_URL}/prices?days=30", timeout=10)
        r_prices.raise_for_status()
        prices = r_prices.json()
        app.logger.info("[DEBUG] /prices raw: %s", prices)
        if 'data' in prices:
            prices = convert_to_col_dict(prices)
            app.logger.info("[DEBUG] /prices converted: %s", prices)
    except requests.exceptions.RequestException as e:
        app.logger.error("Erro ao acessar API /prices: %s", e)
        prices = {'error': f'Erro ao acessar API /prices: {e}'}
    except Exception as e:
        app.logger.error("Erro ao processar dados de /prices: %s", e)
        prices = {'error': f'Erro ao processar dados: {e}'}
    
    # Se tudo falhar, tentar inicializar o banco e treinar o modelo via API
    if pred.get('error') and hist.get('error') and prices.get('error'):
        try:
            app.logger.warning("Tentando inicializar o backend via /api/refresh...")
            r_refresh = requests.post(f"{API_URL}/refresh?force=true", timeout=30)
            r_refresh.raise_for_status()
            app.logger.info("Backend inicializado com sucesso: %s", r_refresh.json())
        except Exception as e:
            app.logger.error("Falha ao inicializar backend: %s", e)
    
    return render_template("index.html", pred={'data': pred}, hist={'data': hist}, prices={'data': prices})

@app.route("/clear_predictions")
def clear_predictions():
    """Rota para limpar as projeções de cotação e redirecionar para a página principal."""
    message = ""
    try:
        app.logger.info("Limpando tabela de projeções via API")
        response = requests.post(f"{API_URL}/clear_predictions", timeout=10)
        response.raise_for_status()
        app.logger.info("Tabela de projeções limpa com sucesso")
        message = "Projeções de cotação limpas com sucesso!"
    except Exception as e:
        app.logger.error(f"Erro ao limpar projeções: {e}")
        message = f"Erro ao limpar projeções: {e}"
    
    # Redirecionar para a página principal com mensagem
    # Redirecione para a página principal
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
