# Bitcoin Guru Experiment - Produtização

## Como funciona

- **backend**: API FastAPI, coleta diária do preço do Bitcoin, gera features, treina modelo, expõe endpoints `/api`.
- **frontend**: Flask, exibe previsão e histórico, gráficos com Chart.js e visual moderno com Bootstrap 5.
- **Orquestração**: Docker Compose, Traefik, rede `dokploy-network`, volume `db_data` para persistência SQLite.

## Subir o sistema

1. Certifique-se que a rede `dokploy-network` já existe:
   ```bash
   docker network create dokploy-network || true
   ```
2. Suba os serviços:
   ```bash
   docker-compose up --build -d
   ```

- Backend: https://bitcoinguru.ml.caiosaldanha.com/api
- Frontend: https://bitcoinguru.ml.caiosaldanha.com/

## Endpoints principais

- `POST /api/refresh` — força coleta e re-treinamento
- `GET /api/predict` — previsão baseada no último dado
- `GET /api/history?limit=N` — histórico de previsões

## Volumes
- `db_data` — persiste o banco SQLite

## Observações
- O Traefik deve estar configurado na infraestrutura Dokploy para rotear os domínios corretamente.
- O backend salva o modelo em `backend/app/models/btc_linreg.pkl`.
