"""
Weather app URLs routing.
"""

from django.urls import path, include

urlpatterns = [
    path('', include('weather.api.urls')),
]
