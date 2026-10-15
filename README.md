# WeatherCast AI ⚡

> **Production-Grade 24-Hour AI Predictive Weather Intelligence System for Cairo, Egypt.**  
> Powered by an Extreme Gradient Boosting (XGBoost) Regressor, Django REST Backend, and a 60 FPS Glassmorphic Analytics Dashboard.

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.0%2B-092E20?logo=django&logoColor=white)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20Regressor-EB7236?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Open-Meteo](https://img.shields.io/badge/Data-Open--Meteo%20API-4A90E2)](https://open-meteo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 62 Passing](https://img.shields.io/badge/Tests-62%20Passed-brightgreen.svg)]()

---

## 📌 Executive Summary

**WeatherCast AI** is an operational machine learning forecasting system that delivers genuine, localized 24-hour ahead ambient air temperature predictions (`temperature_2m`) for **Cairo, Egypt** ($t + 24\text{h}$).

Unlike typical toy demos that merely echo third-party forecast feeds, WeatherCast AI executes a **locally-stored, 21-feature serialized XGBoost model**. External APIs (Open-Meteo) are utilized solely as observational sensors for real-time and historical inputs ($\le t$), guaranteeing 100% independent AI inference.

---

## 🌟 Key Highlights & Engineering Features

* 🧠 **Independent 21-Feature Machine Learning Model:**
  * Trained on 10 years of hourly Cairo meteorological observations (87,600 records).
  * **Residual Delta Formulation:** Predicts $\Delta_{24} = T(t+24\text{h}) - T(t-24\text{h})$ anchored to historical diurnal cycles, reconstructed via $T(t+24\text{h}) = \text{lag\_24} + \Delta_{24}$.
  * **Verified Quality Metrics:** Measured holdout **$\text{MAE} = 1.35^\circ\text{C}$**, **$\text{RMSE} = 1.85^\circ\text{C}$**, and **$R^2 = 0.9439$** on unseen data.
* 🛡️ **Zero Temporal Data Leakage:**
  * Strict causal feature extraction ($[t-168\text{h} \dots t-1\text{h}]$ temperature lags, rolling window statistics, and diurnal peak tracking).
  * Embargoed chronological train/test split with an isolated 24-hour purge boundary.
  * Purged walk-forward cross-validation (`gap=24`) preventing any target contamination.
* ⚡ **High-Performance Glassmorphic UI:**
  * 3D translucent Glassmorphism dashboard with zero browser input spinners.
  * Central hero circular gauge with real-time temperature delta and forecast horizon.
  * 6 interactive Exploratory Data Analysis (EDA) deep-dive charts with plain-English insights.
  * GPU-accelerated smooth scrolling (60 FPS) and full English localization.
* 🔒 **Hardened Django REST Backend:**
  * Thread-safe model singleton with sub-millisecond inference latency ($< 0.5\text{ ms}$).
  * Resilient error handling, coordinate bounding, and timeout-protected API fallbacks.
  * 62 automated unit, integration, parity, and security tests passing with 100% success.

---

## 🏗️ System Architecture

```text
WeatherCast-AI/
├── backend/                       # Production Django REST Framework Application
│   ├── config/settings/           # Base, Development, and Production settings
│   ├── weather/                   # Core Weather & Prediction Engine
│   │   ├── api/                   # REST Views (/api/predict/, /api/forecast/, /api/health/)
│   │   ├── domain/                # Physical Contracts, Schemas, Calibration Constants
│   │   ├── services/              # ModelService, FeatureService, ForecastService, WeatherService
│   │   └── tests/                 # 62 Automated Unit, Integration, & Integrity Tests
│   └── manage.py
├── frontend/                      # Responsive Glassmorphic Dashboard
│   ├── index.html                 # Modern Translucent Dashboard UI
│   └── assets/                    # Optimized Styles, Scripts & Baseline Data
│       ├── css/app.css            # Dark Glassmorphism Design System (Zero Lag)
│       ├── js/app.js              # Operational API Client & Chart Orchestration
│       └── js/eda_data.js         # Compiled 10-Year Cairo ERA5 Historical Analytics
├── ml/                            # Machine Learning Pipeline & Artifacts
│   ├── data/weather.csv           # 10-Year Cairo Hourly Meteorological Dataset
│   ├── models/                    # Serialized Model Artifacts & Metadata
│   │   ├── xgboost_weather_model.json   # Active 21-Feature XGBoost Model
│   │   ├── feature_config.json          # Frozen Production Feature Schema
│   │   └── model_metadata.json          # Checksums, Hyperparameters & Provenance
│   └── training/                  # train.py, features.py, evaluation.py, validation.py
├── scripts/                       # Verification, Parity & Audit Suites
│   ├── verify_model.py            # Model serialization & inference sanity check
│   ├── evaluate_realworld_30d.py  # 61-day real-world operational evaluation
│   └── e2e_test.py                # End-to-end routing & API test
├── docs/                          # Engineering Reports & Audit Logs
│   └── FINAL_ENGINEERING_VERIFICATION.md
├── Dockerfile                     # Optimized Container Runtime
├── docker-compose.yml             # Local Multi-Container Orchestration
└── README.md
```

---

## 📊 Model Performance & Benchmarks

Evaluated on **17,505 unseen chronological test rows** (held-out out-of-sample split):

| Metric | Measured Holdout Result | Baseline Persistence (Lag-24) | Improvement |
|:---|:---:|:---:|:---:|
| **Mean Absolute Error (MAE)** | **$1.3516^\circ\text{C}$** | $2.3140^\circ\text{C}$ | **$41.6\%$ Error Reduction** |
| **Root Mean Squared Error (RMSE)** | **$1.8510^\circ\text{C}$** | $3.1850^\circ\text{C}$ | **$41.9\%$ Improvement** |
| **Coefficient of Determination ($R^2$)** | **$0.9439$** | $0.8340$ | **High Explanatory Power** |
| **Median Absolute Error (MedAE)** | **$1.0149^\circ\text{C}$** | $1.7200^\circ\text{C}$ | **$41.0\%$ Error Reduction** |

---

## 🚀 Quickstart & Installation

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/mohamed-hany74/AI-Weather-Forecasting.git
cd AI-Weather-Forecasting

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 2. Run Automated Verification Tests
```bash
# Run all 62 Django backend and model integrity tests
python backend/manage.py test weather

# Run model inference sanity verification
python scripts/verify_model.py
```

### 3. Launch the Application
```bash
python backend/manage.py runserver 127.0.0.1:8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser to access the dashboard.

---

## 📡 API Reference

### `POST /api/forecast/`
Easy-input endpoint for Cairo. Gathers verified observations and executes production inference.
```json
// Request Body:
{
  "latitude": 30.0444,
  "longitude": 31.2357
}

// Response (200 OK):
{
  "status": "success",
  "prediction": {
    "target_variable": "temperature_2m",
    "forecast_horizon_hours": 24,
    "forecast_time": "2026-10-16T03:00:00Z",
    "predicted_temperature_2m": 21.29,
    "current_temperature": 20.1,
    "change_vs_current": 1.19,
    "typical_error_celsius": 1.4,
    "unit": "°C"
  }
}
```

### `POST /api/predict/`
Custom manual prediction endpoint allowing full user control over atmospheric inputs.

---

## 👤 Author & Maintainer

* **Developer:** Mohamed Hany ([@mohamed-hany74](https://github.com/mohamed-hany74))
* **Email:** mhmdhanyx@gmail.com
* **Repository:** [https://github.com/mohamed-hany74/AI-Weather-Forecasting](https://github.com/mohamed-hany74/AI-Weather-Forecasting)
