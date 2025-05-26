from fastapi import FastAPI, HTTPException, Query, APIRouter
from fastapi.responses import JSONResponse
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import requests
import joblib
import os
from datetime import datetime, timezone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DB_PATH = '/data/db.sqlite'
MODEL_PATH = 'models/btc_linreg.pkl'

router = APIRouter(prefix="/api")
app = FastAPI()
scheduler = BackgroundScheduler()
engine = create_engine(f'sqlite:///{DB_PATH}', connect_args={"check_same_thread": False})

# --- DB INIT ---
def init_db():
    with engine.begin() as conn:
        conn.execute(text('''CREATE TABLE IF NOT EXISTS btc_data (
            date TEXT PRIMARY KEY,
            price REAL,
            lag_1 REAL, lag_2 REAL, lag_3 REAL, lag_4 REAL, lag_5 REAL, lag_6 REAL, lag_7 REAL,
            ma_7 REAL, ma_14 REAL,
            ret_1d REAL, ret_7d REAL,
            dow INTEGER
        )'''))
        conn.execute(text('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            pred_7d REAL
        )'''))

init_db()

# --- Feature Engineering ---
def make_features(df):
    for lag in range(1, 8):
        df[f'lag_{lag}'] = df['price'].shift(lag)
    df['ma_7'] = df['price'].rolling(7).mean()
    df['ma_14'] = df['price'].rolling(14).mean()
    df['ret_1d'] = df['price'].pct_change()
    df['ret_7d'] = df['price'].pct_change(7)
    df['dow'] = df['date'].dt.dayofweek
    return df

# --- Data Fetch & Insert ---
def fetch_and_insert(force=False):
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=1'
    r = requests.get(url, timeout=10)
    data = r.json()
    if 'prices' not in data:
        raise HTTPException(502, 'CoinGecko API error')
    ts, price = data['prices'][-1]
    date = datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d')
    with engine.begin() as conn:
        exists = conn.execute(text('SELECT 1 FROM btc_data WHERE date=:date'), {'date': date}).fetchone()
        if exists and not force:
            return False
        # Get last 30 days for features
        df = pd.read_sql('SELECT * FROM btc_data ORDER BY date DESC LIMIT 30', conn)
        df = df.sort_values('date')
        new_row = pd.DataFrame({'date': [date], 'price': [price]})
        df = pd.concat([df, new_row], ignore_index=True)
        df['date'] = pd.to_datetime(df['date'])
        df = make_features(df)
        row = df.iloc[-1]
        # Insert
        conn.execute(text('''INSERT OR REPLACE INTO btc_data (date, price, lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, ma_7, ma_14, ret_1d, ret_7d, dow) VALUES (:date, :price, :lag_1, :lag_2, :lag_3, :lag_4, :lag_5, :lag_6, :lag_7, :ma_7, :ma_14, :ret_1d, :ret_7d, :dow)'''), row.to_dict())
    retrain_model()
    return True

# --- Model Training ---
def retrain_model():
    with engine.begin() as conn:
        df = pd.read_sql('SELECT * FROM btc_data ORDER BY date', conn)
    df = df.dropna()
    FEATURES = [f'lag_{i}' for i in range(1,8)] + ['ma_7','ma_14','ret_1d','ret_7d','dow']
    X = df[FEATURES].values
    y = df['price'].values
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({'model': model, 'features': FEATURES}, MODEL_PATH)

# --- Scheduler ---
def scheduled_job():
    try:
        fetch_and_insert()
    except Exception as e:
        print('Scheduled job failed:', e)

scheduler.add_job(scheduled_job, 'cron', hour=0, minute=15)
scheduler.start()

# --- API Endpoints ---
@router.post('/refresh')
def api_refresh(force: bool = Query(False)):
    changed = fetch_and_insert(force=force)
    if not changed:
        return JSONResponse(status_code=204, content={'detail': 'Already up to date'})
    return {'detail': 'Data refreshed and model retrained'}

@router.get('/predict')
def api_predict():
    with engine.begin() as conn:
        df = pd.read_sql('SELECT * FROM btc_data ORDER BY date DESC LIMIT 30', conn).sort_values('date')
    if df.empty:
        raise HTTPException(404, 'No data')
    df['date'] = pd.to_datetime(df['date'])
    df = make_features(df)
    row = df.dropna().iloc[-1]
    # Load model
    model_obj = joblib.load(MODEL_PATH)
    model, FEATURES = model_obj['model'], model_obj['features']
    X = row[FEATURES].values.reshape(1, -1)
    pred_7d = float(model.predict(X)[0])
    # Metrics
    with engine.begin() as conn:
        df_train = pd.read_sql('SELECT * FROM btc_data ORDER BY date', conn).dropna()
    X_train = df_train[FEATURES].values
    y_train = df_train['price'].values
    y_pred = model.predict(X_train)
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = float(mean_absolute_error(y_train, y_pred))
    r2 = float(r2_score(y_train, y_pred))
    # Save prediction
    with engine.begin() as conn:
        conn.execute(text('INSERT INTO predictions (date, pred_7d) VALUES (:date, :pred_7d)'), {'date': row['date'].strftime('%Y-%m-%d'), 'pred_7d': pred_7d})
    # Response
    out = pd.DataFrame({
        'date': [row['date'].strftime('%Y-%m-%d')],
        'price_now': [row['price']],
        'pred_7d': [pred_7d],
        'mae_train': [mae],
        'r2_train': [r2]
    })
    return JSONResponse(content=out.to_dict(orient='split'))

@router.get('/history')
def api_history(limit: int = Query(10)):
    with engine.begin() as conn:
        df = pd.read_sql(f'SELECT * FROM predictions ORDER BY run_ts DESC LIMIT {limit}', conn)
    return JSONResponse(content=df.to_dict(orient='split'))

app.include_router(router)
