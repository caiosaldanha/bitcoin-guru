# 🚀 Bitcoin Guru - Seu Consultor IA para Investimentos em Bitcoin

![Bitcoin Guru](https://img.shields.io/badge/Bitcoin-Guru-orange?style=for-the-badge&logo=bitcoin&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

## 📊 O que é o Bitcoin Guru?

O **Bitcoin Guru** é uma aplicação completa de Machine Learning que te ajuda a tomar decisões inteligentes sobre investimentos em Bitcoin! 🎯

Usando algoritmos avançados de regressão linear e análise técnica, nossa IA analisa padrões históricos do Bitcoin para prever o comportamento do preço nos próximos 7 dias. Não é só mais um gráfico - é sua consultoria pessoal disponível 24/7! 💡

**🌐 Acesse agora:** [https://bitcoinguru.ml.caiosaldanha.com/](https://bitcoinguru.ml.caiosaldanha.com/)

---

## 🏗️ Arquitetura do Sistema

Nossa aplicação segue uma arquitetura moderna de microserviços:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   🌐 Frontend   │    │   🔧 Backend    │    │   💾 Database   │
│     Flask       │◄──►│    FastAPI      │◄──►│    SQLite       │
│   Bootstrap 5   │    │   Scheduler     │    │   Persistent    │
│   Chart.js      │    │   ML Pipeline   │    │    Volume       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   🌍 CoinGecko  │
                    │      API        │
                    │  (Dados Reais)  │
                    └─────────────────┘
```

### 🔧 **Backend (FastAPI)**
- **🤖 Engine de IA**: Modelo de regressão linear com pipeline de pré-processamento
- **📈 Coleta de Dados**: Integração automática com CoinGecko API
- **⏰ Scheduler**: Coleta diária automatizada de preços
- **🗄️ Persistência**: SQLite para dados históricos e previsões
- **📊 Feature Engineering**: 15+ indicadores técnicos calculados automaticamente

### 🌐 **Frontend (Flask)**
- **🎨 UI Moderna**: Interface responsiva com Bootstrap 5
- **📊 Dashboards Interativos**: 4 gráficos diferentes com Chart.js
- **📱 Mobile-First**: Funciona perfeitamente em dispositivos móveis
- **⚡ Real-time**: Atualização automática das previsões

---

## 🧠 Como Funciona o Modelo de IA

### 📊 **Features Calculadas (15 indicadores)**

O modelo utiliza **15 features técnicas** para fazer previsões precisas:

| Categoria | Features | Descrição |
|-----------|----------|-----------|
| **🕰️ Lags** | `lag_1` a `lag_7` | Preços dos últimos 7 dias |
| **📈 Médias Móveis** | `ma_7`, `ma_14` | Médias móveis de 7 e 14 dias |
| **🎢 Retornos** | `ret_1d`, `ret_7d` | Variação percentual de 1 e 7 dias |
| **📅 Temporal** | `dow` | Dia da semana (0=Segunda, 6=Domingo) |

### 🎯 **Pipeline de Machine Learning**

```python
# Pipeline otimizado para previsão de preços
modelo = make_pipeline(
    StandardScaler(),        # Normalização dos dados
    Ridge(alpha=1.0)        # Regressão linear com regularização
)
```

**Por que Ridge Regression?**
- ✅ **Robusta**: Resiste bem a overfitting
- ✅ **Rápida**: Treinamento em segundos
- ✅ **Interpretável**: Fácil de entender e debugar
- ✅ **Estável**: Funciona bem com dados financeiros

### 📊 **Métricas de Performance**

O modelo é constantemente avaliado usando:

- **MAE (Mean Absolute Error)**: ~$3,500 💰
- **R² (Coeficiente de Determinação)**: ~91% 🎯
- **Horizon**: 7 dias de previsão 📅

---

## 🚀 API Endpoints

### 🔮 **Previsões**
```http
GET /api/predict
```
**Retorna**: Previsão para os próximos 7 dias + métricas do modelo

**Exemplo de resposta**:
```json
{
  "data": [
    ["2025-05-27", "2025-06-03", 108540.21, 108718.65, 3894.26, 0.91]
  ],
  "columns": ["date", "forecast_date", "price_now", "pred_7d", "mae_train", "r2_train"]
}
```

### 📊 **Dados Técnicos**
```http
GET /api/technical_data?days=30
```
**Retorna**: Dados para dashboards (preços, médias móveis, volatilidade, performance)

### 📈 **Histórico de Preços**
```http
GET /api/prices?days=30
```
**Retorna**: Preços históricos dos últimos N dias

### 🕒 **Histórico de Previsões**
```http
GET /api/history?limit=10
```
**Retorna**: Últimas previsões feitas pelo modelo

### 🔄 **Operações de Manutenção**
```http
POST /api/refresh?force=true    # Forçar atualização de dados
POST /api/clear_predictions     # Limpar histórico de previsões
GET  /api/dbdump               # Debug: visualizar todos os dados
```

---

## 🎨 Interface & Dashboards

### 🏠 **Dashboard Principal**

![Dashboard](https://via.placeholder.com/800x400/1e293b/ffffff?text=🏠+Dashboard+Principal)

**Elementos principais:**
- 💰 **Preço Atual vs Projetado**: Comparação visual com variação percentual
- 🎯 **Recomendação IA**: COMPRAR/VENDER/MANTER baseada na previsão
- 📊 **Métricas do Modelo**: MAE e R² em tempo real
- 📅 **Data da Análise**: Timestamp da última previsão

### 📊 **Dashboard de Performance**

![Performance](https://via.placeholder.com/800x300/059669/ffffff?text=📊+Performance+do+Modelo+IA)

**Gráfico de Performance do Modelo IA**
- 📈 **Linha Vermelha**: MAE (Erro Médio) em dólares
- 📈 **Linha Verde**: R² (Precisão) em percentual  
- 📅 **Período**: Últimos 14 dias de evolução
- 🎯 **Interpretação**: Quanto menor o MAE e maior o R², melhor o modelo

### 🌪️ **Dashboard de Volatilidade**

![Volatilidade](https://via.placeholder.com/800x300/dc2626/ffffff?text=🌪️+Análise+de+Volatilidade)

**Gráfico de Volatilidade**
- 📊 **Barras Coloridas**: Volatilidade diária (verde=baixa, amarelo=média, vermelho=alta)
- 📉 **Índice Geral**: Volatilidade média dos últimos 7 dias
- ⚠️ **Alert System**: Cores indicam nível de risco

### 📈 **Dashboard de Análise Técnica**

![Análise Técnica](https://via.placeholder.com/800x400/3b82f6/ffffff?text=📈+Análise+Técnica+Completa)

**Gráfico Técnico Completo**
- 🔵 **Linha Azul**: Preços históricos (30 dias)
- 🟢 **Linha Verde Clara**: Média móvel de 7 dias
- 🟢 **Linha Verde Escura**: Média móvel de 14 dias  
- 🔴 **Ponto Vermelho**: Previsão para +7 dias
- 📊 **Barras de Fundo**: Indicador de volatilidade

### 💹 **Dashboard de Cotações**

![Cotações](https://via.placeholder.com/800x300/8b5cf6/ffffff?text=💹+Cotações+dos+Últimos+30+Dias)

**Gráfico de Preços Históricos**
- 📊 **Linha Suave**: Evolução dos preços  
- 🎨 **Gradiente**: Preenchimento visual
- 🎯 **Interativo**: Hover mostra preço exato

---

## 🛠️ Tecnologias Utilizadas

### 🔙 **Backend Stack**
- ![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white) **Python 3.11**
- ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white) **FastAPI** - API moderna e rápida
- ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) **Pandas** - Manipulação de dados
- ![Scikit-learn](https://img.shields.io/badge/Sklearn-F7931E?logo=scikit-learn&logoColor=white) **Scikit-learn** - Machine Learning
- ![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white) **SQLite** - Banco de dados
- ![APScheduler](https://img.shields.io/badge/APScheduler-FF6B35?logoColor=white) **APScheduler** - Tarefas agendadas

### 🎨 **Frontend Stack**
- ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) **Flask** - Web framework
- ![Bootstrap](https://img.shields.io/badge/Bootstrap_5-7952B3?logo=bootstrap&logoColor=white) **Bootstrap 5** - UI framework
- ![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chart.js&logoColor=white) **Chart.js** - Gráficos interativos
- ![FontAwesome](https://img.shields.io/badge/FontAwesome-339AF0?logo=fontawesome&logoColor=white) **FontAwesome** - Ícones

### 🚀 **Deploy & Infrastructure**
- ![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white) **Docker & Docker Compose**
- ![Traefik](https://img.shields.io/badge/Traefik-24A1C1?logo=traefik&logoColor=white) **Traefik** - Reverse proxy & SSL
- ![Dokploy](https://img.shields.io/badge/Dokploy-FF6B35?logoColor=white) **Dokploy** - Deploy automatizado

---

## 🚀 Como Rodar o Projeto

### 🐳 **Deploy Completo (Recomendado)**

```bash
# 1. Clone o repositório
git clone https://github.com/caiosaldanha/bitcoin-guru.git
cd bitcoin-guru

# 2. Crie a rede Docker (se não existir)
docker network create dokploy-network || true

# 3. Suba os serviços
docker-compose up --build -d

# 4. Acesse a aplicação
# Frontend: https://bitcoinguru.ml.caiosaldanha.com/
# Backend:  https://bitcoinguru.ml.caiosaldanha.com/api/docs
```

### 🔧 **Desenvolvimento Local**

```bash
# Backend
cd backend/app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (novo terminal)
cd frontend/app
pip install -r requirements.txt
export API_URL=http://localhost:8000/api
python main.py
```

---

## 📁 Estrutura do Projeto

```
bitcoin-guru/
├── 📁 backend/                 # API FastAPI
│   ├── 🐳 Dockerfile
│   └── 📁 app/
│       ├── 🐍 main.py         # Core da aplicação + ML
│       ├── 📋 requirements.txt
│       └── 📁 models/         # Modelos treinados (.pkl)
├── 📁 frontend/               # Interface Flask  
│   ├── 🐳 Dockerfile
│   └── 📁 app/
│       ├── 🐍 main.py         # Web server
│       ├── 📋 requirements.txt
│       └── 📁 templates/
│           └── 🌐 index.html  # Interface principal
├── 🐳 docker-compose.yml     # Orquestração
├── 📓 README.md              # Este arquivo! 
└── 🧪 fase3_ml_regressao_linear_v1.ipynb  # Notebook original
```

---

## 🎯 Funcionalidades Especiais

### 🤖 **IA Auto-Treinante**
- ✅ **Bootstrap Automático**: Carrega 365 dias de dados históricos na primeira execução
- ✅ **Atualização Diária**: Coleta novos dados automaticamente
- ✅ **Re-treino Inteligente**: Modelo se atualiza quando necessário
- ✅ **Fallback Robusto**: Se algo falha, a aplicação se auto-recupera

### 📊 **Dashboards Inteligentes** 
- ✅ **Dados Sintéticos**: Se não há histórico suficiente, gera dados realistas para visualização
- ✅ **Responsivo**: Funciona perfeitamente em mobile, tablet e desktop
- ✅ **Tempo Real**: Métricas atualizadas automaticamente
- ✅ **Interativo**: Hover, zoom, tooltips informativos

### 🔧 **Ferramentas de Debug**
- ✅ **Botão de Limpeza**: Remove histórico de previsões (canto inferior esquerdo)
- ✅ **Endpoint de Debug**: `/api/dbdump` para ver todos os dados
- ✅ **Logs Detalhados**: Sistema completo de logging
- ✅ **Métricas Expostas**: Performance do modelo visível na interface

---

## 📈 Interpretando os Resultados

### 🎯 **Recomendações**

| Variação | Recomendação | Significado |
|----------|--------------|-------------|
| **> +2%** | 🟢 **COMPRAR** | Tendência de alta forte |
| **-2% a +2%** | 🟡 **MANTER** | Preço estável, mercado lateral |
| **< -2%** | 🔴 **VENDER** | Tendência de baixa |

### 📊 **Métricas do Modelo**

- **MAE ~$3,500**: O modelo erra em média $3,500 (excelente para Bitcoin!)
- **R² ~91%**: O modelo explica 91% da variação dos preços
- **Horizon 7d**: Previsões para uma semana à frente

### 🌪️ **Índice de Volatilidade**

| Valor | Cor | Interpretação |
|-------|-----|---------------|
| **< 3%** | 🟢 Verde | Baixa volatilidade, movimento calmo |
| **3-6%** | 🟡 Amarelo | Volatilidade média, movimento normal |
| **> 6%** | 🔴 Vermelho | Alta volatilidade, movimento agitado |

---

## 🤝 Contribuindo

Adoraríamos sua contribuição! 🎉

1. 🍴 Faça um fork do projeto
2. 🌟 Crie sua feature branch (`git checkout -b feature/MinhaFeature`)
3. 💾 Commit suas mudanças (`git commit -m 'Adiciona MinhaFeature'`)
4. 📤 Push para a branch (`git push origin feature/MinhaFeature`)
5. 🔀 Abra um Pull Request

---

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

---

## 👨‍💻 Desenvolvido com ❤️ por

**Caio Saldanha** - [GitHub](https://github.com/caiosaldanha)

---

## 🚨 Disclaimer

⚠️ **Importante**: Este projeto é para fins educacionais e demonstração de Machine Learning. Não constitui consultoria financeira. Sempre faça sua própria pesquisa antes de investir!

📊 **Dados**: Fornecidos pela CoinGecko API  
🤖 **IA**: Ridge Regression com 15 features técnicas  
📅 **Atualização**: Dados e previsões atualizados automaticamente

---

<div align="center">

### 🌟 Se este projeto te ajudou, deixe uma estrela! ⭐

[![Star](https://img.shields.io/github/stars/caiosaldanha/bitcoin-guru?style=social)](https://github.com/caiosaldanha/bitcoin-guru)

</div>
