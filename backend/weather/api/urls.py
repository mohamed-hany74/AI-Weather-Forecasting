"""
API URLs routing for weather endpoints.
"""

from django.urls import path
from .views import health_check, readiness_check, model_info, predict_temperature_view, forecast_easy_view

urlpatterns = [
    path('health/', health_check, name='api-health'),
    path('ready/', readiness_check, name='api-ready'),
    path('model-info/', model_info, name='api-model-info'),
    path('predict/', predict_temperature_view, name='api-predict'),
    path('forecast/', forecast_easy_view, name='api-forecast'),
]

