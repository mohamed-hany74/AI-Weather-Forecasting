# 🌤️ WeatherCast AI

### Enterprise 24-Hour Predictive Weather Intelligence Platform

> Production-grade Machine Learning system for forecasting 2-meter air temperature in Cairo, Egypt exactly 24 hours into the future.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-REST-green.svg)](https://www.djangoproject.com/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Supported-blue.svg)](https://www.docker.com/)
[![Security](https://img.shields.io/badge/Security-0%20Secrets%20Detected-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-16%2F16%20Passed-brightgreen.svg)]()
[![Model](https://img.shields.io/badge/Model-R%C2%B2%200.9400-brightgreen.svg)]()

---

## 🚀 Overview

**WeatherCast AI** is an end-to-end Machine Learning forecasting platform designed to predict Cairo's ambient 2-meter air temperature (`temperature_2m`) **24 hours ahead**.

The system combines:

- Historical temperature observations
- 168 hours of causal historical context
- Atmospheric measurements
- Cyclical temporal features
- XGBoost regression
- Django REST API
- Open-Meteo weather services
- Production feature engineering
- Historical backtesting
- Automated parity verification
- Security scanning
- Docker deployment
- Responsive web dashboard

The project was engineered with a strong focus on **production reliability, reproducibility, causal feature engineering, and train-to-production consistency**.

---

# 🎯 Problem Definition

Given the current time:

```text
t
