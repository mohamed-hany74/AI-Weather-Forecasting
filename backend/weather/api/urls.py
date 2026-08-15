"""
API URLs routing for weather endpoints.
"""

from django.urls import path
from .views import health_check, model_info, predict_temperature_view

urlpatterns = [
    path('health/', health_check, name='api-health'),
    path('model-info/', model_info, name='api-model-info'),
    path('predict/', predict_temperature_view, name='api-predict'),
]
