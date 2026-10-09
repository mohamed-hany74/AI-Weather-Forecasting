import os
import sys
import logging
from django.apps import AppConfig

logger = logging.getLogger(__name__)


class WeatherConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'weather'

    def ready(self):
        """Preload ML model artifact into memory upon application initialization."""
        # Avoid running during management commands like makemigrations, migrate, etc.
        if 'runserver' in sys.argv or 'gunicorn' in sys.argv[0] or 'uvicorn' in sys.argv[0] or os.environ.get('RUN_MAIN') == 'true':
            try:
                from .services.model_service import load_model_artifacts
                load_model_artifacts()
                logger.info("ML Model preloaded successfully at startup.")
            except Exception as e:
                logger.warning(f"Could not preload model at startup: {e}")

