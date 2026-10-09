"""
Production settings for WeatherCast AI.
"""

import os
from .base import *

DEBUG = False
ALLOWED_HOSTS = os.environ.get('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')
SECRET_KEY = os.environ['DJANGO_SECRET_KEY']

# Security headers
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'
SECURE_CONTENT_TYPE_NOSNIFF = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

# Conditional Production SSL & HSTS Hardening
ENABLE_SSL = os.environ.get('DJANGO_ENABLE_SSL', 'false').lower() in ('true', '1', 'yes')
if ENABLE_SSL:
    SECURE_SSL_REDIRECT = True
    SECURE_HSTS_SECONDS = 31536000  # 1 year
    SECURE_HSTS_INCLUDE_SUBDOMAINS = True
    SECURE_HSTS_PRELOAD = True
    SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
else:
    SECURE_SSL_REDIRECT = False
    SECURE_HSTS_SECONDS = 0

