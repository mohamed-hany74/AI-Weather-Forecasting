"""
URL Configuration for WeatherCast AI.
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.views.static import serve
from django.urls import path, include, re_path
from django.views.generic import TemplateView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('weather.api.urls')),
    path('', TemplateView.as_view(template_name='index.html'), name='dashboard'),
    re_path(r'^assets/(?P<path>.*)$', serve, {'document_root': settings.FRONTEND_DIR / 'assets'}),
    re_path(r'^static/(?P<path>.*)$', serve, {'document_root': settings.FRONTEND_DIR / 'assets'}),
]

if settings.DEBUG and settings.STATICFILES_DIRS:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])
