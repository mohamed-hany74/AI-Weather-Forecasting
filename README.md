# WeatherCast AI ⚡

**Enterprise 24-Hour Predictive Weather Intelligence Platform for Cairo, Egypt**  
*Production Machine Learning Forecasting Pipeline with XGBoost, Django REST API, and Vanilla JavaScript SaaS Dashboard.*

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Django 5/6](https://img.shields.io/badge/Django-REST-green.svg)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-orange.svg)](https://xgboost.readthedocs.io/)
[![Security: Clean](https://img.shields.io/badge/Security-0%20Secrets%20Found-brightgreen.svg)]()

---

## 🌟 Executive Overview

**WeatherCast AI** is a production-grade machine learning platform designed to forecast ambient 2-meter air temperature (`temperature_2m`) in Cairo, Egypt exactly 24 hours into the future ($t + 24\text{h}$). 

The system relies strictly on a **causal 168-hour historical context** ($t - 168\text{h} \dots t - 1\text{h}$) and real-time atmospheric measurements from Open-Meteo at prediction instant $t$. It guarantees **zero lookahead bias**, zero future target leakage, and 100% mathematical parity between training equations and production inference services.

---

## 🏗️ System Architecture

```text
                               ┌────────────────────────────────┐
                               │   Vanilla JS / CSS Dashboard   │
                               │   (168h Slicing & Visualizer)  │
                               └───────────────┬────────────────┘
                                               │ HTTP / JSON
                                               ▼
                               ┌────────────────────────────────┐
                               │      Django REST API Layer     │
                               │      (/api/predict/, /health/) │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               │                               │                               │
               ▼                               ▼                               ▼
     ┌──────────────────┐            ┌───────────────────┐           ┌──────────────────┐
     │ Validation Layer │            │  Weather Service  │           │ Feature Service  │
     │  (Schema Checks) │            │ (Open-Meteo Sync) │           │(15 Causal Vector)│
     └──────────────────┘            └─────────┬─────────┘           └─────────┬────────┘
                                               │ Atmospheric at t              │
                                               └───────────────┬───────────────┘
                                                               │ 15-Feature Array
                                                               ▼
                                                   ┌──────────────────────┐
                                                   │    Model Service     │
                                                   │ (XGBoost Singleton)  │
                                                   └───────────┬──────────┘
                                                               │
                                                               ▼
                                                   ┌──────────────────────┐
                                                   │  Forecast (t + 24h)  │
                                                   └──────────────────────┘
```

---

## 📊 Measured Model Performance & Baselines

Evaluated on **17,493 untouched out-of-sample chronological test rows** (2018–2019):

| Forecasting Strategy / Model | MAE (°C) | RMSE (°C) | $R^2$ Score | MedAE (°C) | Improvement vs Strategy |
|:---|:---:|:---:|:---:|:---:|:---:|
| **XGBoost Production Model** | **1.3652** | **1.9144** | **0.9400** | **0.9725** | **Baseline Anchor** |
| Baseline 1: 1-hour Persistence ($t-1\text{h}$) | 1.8497 | 2.4740 | 0.8998 | 1.4500 | **+26.2% better** |
| Baseline 2: 24-hour Persistence ($t-24\text{h}$) | 1.8909 | 2.6628 | 0.8839 | 1.4800 | **+27.8% better** |
| Baseline 3: 24-hour Rolling Mean | 4.1290 | 4.8884 | 0.6089 | 3.5100 | **+66.9% better** |
| Baseline 4: 7-day Rolling Mean | 4.2320 | 5.0563 | 0.5816 | 3.6200 | **+67.7% better** |

---

## 📐 15-Feature Causal Specification

The model consumes a frozen 15-element feature vector in this exact sequence:

1. `apparent_temperature`: Atmospheric feel at $t$ via Open-Meteo.
2. `pressure_msl`: Mean sea level pressure at $t$ (hPa).
3. `relative_humidity_2m`: Air saturation at $t$ (%).
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

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/ToqaGhozlan/Machine_Learning_Summer_Intern_R5.git
cd Machine_Learning_Summer_Intern_R5

# Create & activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env and supply your secret key
```

### 3. Run Automated Tests & Verifications

```bash
# Run Django test suite
python backend/manage.py test weather -v 2

# Verify feature engineering parity (< 1e-6 diff)
python scripts/verify_features.py

# Verify model serialization
python scripts/verify_model.py

# Run comprehensive smoke test
python scripts/smoke_test.py
```

### 4. Start Development Server

```bash
python backend/manage.py runserver 127.0.0.1:8000
```
Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** to view the live dashboard.

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```
The application will be accessible at `http://localhost:8000`.

---

## 🔒 Security & Best Practices

- **Zero Hardcoded Secrets:** All secrets, keys, and environments are loaded via environment variables (`.env`).
- **Input Sanitization:** All payload arrays and timestamps undergo strict bounds checking, shape assertions, and UTC normalization.
- **Fail-Safe Fallbacks:** External Open-Meteo requests timeout after 10 seconds and return structured HTTP 503 error payloads without exposing internal tracebacks.

---

## 📄 License

This project is licensed under the MIT License.
