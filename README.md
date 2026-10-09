# WeatherCast AI ⚡

> **Production-Grade 24-Hour Predictive Weather Intelligence System for Cairo, Egypt.**  
> Powered by an XGBoost Regressor, Django REST API, and a Real-Time Vanilla JS Analytics Dashboard.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0%2B-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-EB7236)](https://xgboost.readthedocs.io/)
[![Open-Meteo](https://img.shields.io/badge/Data-Open--Meteo%20API-4A90E2)](https://open-meteo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

**WeatherCast AI** is an end-to-end machine learning system engineered to predict ambient 2-meter air temperature (`temperature_2m`) in Cairo, Egypt **24 hours ahead** ($t + 24\text{h}$).

### Key Principles
- **Strictly Causal 168-Hour Window:** Consumes strictly $[t-168\text{h} \dots t-1\text{h}]$ observed history: no look-ahead in features; imputation after split; test period not used for training.
- **Real-Time Atmospheric Sync:** Integrates with the Open-Meteo API to fetch real-time and historical exogenous variables at reference timestamp $t$.
- **Mathematical Feature Parity:** 100% parity between training equations and production Django services ($\text{Max Error} < 10^{-11}$).
- **Thread-Safe Model Singleton:** The serialized XGBoost booster JSON model is loaded once into memory on server boot ($0.38\text{ ms}$ inference latency).

---

## 🏗️ Architecture

```text
WeatherCast-AI/
├── backend/                   # Django REST Framework & Domain Layer
│   ├── config/settings/       # Base, Development, and Production settings
│   ├── weather/               # Core application
│   │   ├── api/               # Serializers, Views (/predict, /health, /model-info)
│   │   ├── domain/            # Schemas, Contracts, Custom Exceptions
│   │   ├── services/          # ModelService, FeatureService, WeatherService
│   │   └── tests/             # 16 Unit & Integration tests
│   └── manage.py
├── frontend/                  # Responsive Dark SaaS Dashboard UI
│   ├── index.html             # Glassmorphism interface with Chart.js
│   └── assets/                # Modular CSS & JS (api.js, charts.js, dashboard.js, validation.js)
├── ml/                        # Machine Learning Lifecycle
│   ├── data/weather.csv       # Cairo Hourly Dataset (2010–2019, 87,600 rows)
│   ├── models/                # Serialized model JSON & feature configuration
│   ├── notebooks/             # Clean reproducible Jupyter notebook
│   └── training/              # train.py, features.py, evaluation.py, validation.py
├── scripts/                   # Automated Verification & Audit Scripts
│   ├── verify_features.py     # Independent analytical feature check
│   ├── verify_model.py        # Model loading & sanity test
│   ├── verify_train_production_parity.py  # 200-timestamp deterministic parity test
│   ├── verify_api_historical_predictions.py # 50-point API historical backtest
│   ├── smoke_test.py          # Live Open-Meteo + API E2E smoke test
│   ├── e2e_test.py            # Routing and template rendering test
│   ├── comprehensive_audit.py # Full 8-phase empirical audit runner
│   └── security_scan.py      # Secret & credential scanner (0 findings)
├── .env.example               # Environment variables template
├── .gitignore                 # Enterprise-grade gitignore
├── docker-compose.yml         # Containerized local orchestration
├── Dockerfile                 # Slim production container
└── DEPLOYMENT.md              # Docker & cloud deployment guide
```

---

## 📊 Model Evaluation & Baseline Comparison

Evaluated on **17,493 unseen out-of-sample chronological test rows** (Jan 1, 2018 – Dec 30, 2019):

| Strategy / Model | MAE (°C) | RMSE (°C) | $R^2$ Score | MedAE (°C) | Improvement vs Baseline |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Production Model** | **1.3652** | **1.9144** | **0.9400** | **0.9725** | **Benchmark Anchor** |
| Baseline 1: 1-Hour Persistence ($t-1\text{h}$) | 1.8497 | 2.4740 | 0.8998 | 1.4500 | **+26.2% better** |
| Baseline 2: 24-Hour Persistence ($t-24\text{h}$) | 1.8909 | 2.6628 | 0.8839 | 1.4800 | **+27.8% better** |
| Baseline 3: 24-Hour Rolling Mean | 4.1290 | 4.8884 | 0.6089 | 3.5100 | **+66.9% better** |
| Baseline 4: 7-Day Rolling Mean | 4.2320 | 5.0563 | 0.5816 | 3.6200 | **+67.7% better** |

---

## 📐 The 15 Causal Features

The model consumes a strictly causal 15-element feature vector in this exact order:

1. `apparent_temperature`: Open-Meteo apparent temperature at $t$ (°C).
2. `pressure_msl`: Mean sea level atmospheric pressure at $t$ (hPa).
3. `relative_humidity_2m`: Relative humidity at $t$ (%).
4. `hour_cos`: Cyclical hour $\cos(2\pi \cdot \text{hour}/24)$.
5. `month_cos`: Cyclical month $\cos(2\pi \cdot (\text{month}-1)/12)$.
6. `dayofyear_sin`: Cyclical day of year $\sin(2\pi \cdot (\text{doy}-1)/\text{days})$.
7. `dayofyear_cos`: Cyclical day of year $\cos(2\pi \cdot (\text{doy}-1)/\text{days})$.
8. `temperature_2m_lag_1`: Observed temperature at $t - 1\text{h}$.
9. `temperature_2m_lag_24`: Observed temperature at $t - 24\text{h}$.
10. `temperature_2m_lag_72`: Observed temperature at $t - 72\text{h}$.
11. `temperature_2m_lag_168`: Observed temperature at $t - 168\text{h}$.
12. `temperature_2m_rolling_max_6`: Maximum temperature over $[t-6\text{h} \dots t-1\text{h}]$.
13. `temperature_2m_rolling_max_24`: Maximum temperature over $[t-24\text{h} \dots t-1\text{h}]$.
14. `temperature_2m_rolling_mean_24`: Mean temperature over $[t-24\text{h} \dots t-1\text{h}]$.
15. `temperature_2m_rolling_std_24`: Sample standard deviation over $[t-24\text{h} \dots t-1\text{h}]$ ($\text{ddof}=1$).

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/` | Serves the interactive SaaS forecasting dashboard. |
| `GET` | `/api/health/` | Service health status (`{"status": "ok"}`). |
| `GET` | `/api/model-info/` | Model architecture, metadata, feature order, and test metrics. |
| `POST` | `/api/predict/` | Generates a 24-hour temperature forecast with full feature diagnostics. |

### Example Prediction Request
```bash
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "current_time": "2026-08-15T09:00:00Z",
    "latitude": 30.0444,
    "longitude": 31.2357,
    "temperature_history_168h": [27.5, 27.2, "...", 31.1]
  }'
```

---

## 🚀 Quickstart & Local Setup

### 1. Clone & Install Dependencies
```bash
# Clone repository
git clone https://github.com/ToqaGhozlan/Machine_Learning_Summer_Intern_R5.git
cd Machine_Learning_Summer_Intern_R5

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r backend/requirements.txt
```

### 2. Environment Configuration
```bash
cp .env.example .env
```

### 3. Run Automated Tests & Verifications
```bash
# Django unit and integration test suite (16 tests)
python backend/manage.py test weather -v 2

# Parity & audit verifications
python scripts/verify_train_production_parity.py
python scripts/smoke_test.py
```

### 4. Start the Application
```bash
python backend/manage.py runserver 127.0.0.1:8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🐳 Docker Support

```bash
docker-compose up --build
```

---

## 🛡️ Security & DevSecOps

- **Zero Secret Commits:** Verified by `scripts/security_scan.py` with 0 findings.
- **Environment Isolation:** Secrets loaded exclusively via `.env`.
- **Strict Input Validation:** Rejects NaN, Infinity, incorrect history length ($\neq 168$), and out-of-bound coordinates.
- **Network Resilience:** External API calls wrapped with 10s timeouts and structured HTTP 503 fallback handlers.

---

## 📄 License

This project is licensed under the MIT License.
