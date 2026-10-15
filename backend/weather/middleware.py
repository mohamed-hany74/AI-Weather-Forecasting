"""
CORS Middleware for WeatherCast AI.
Allows requests from frontend development servers and origins.
"""
from django.http import HttpResponse


class CorsMiddleware:
    """Simple CORS middleware allowing cross-origin requests to API."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.method == "OPTIONS":
            response = HttpResponse()
            self._add_cors_headers(response)
            return response

        response = self.get_response(request)
        self._add_cors_headers(response)
        return response

    @staticmethod
    def _add_cors_headers(response):
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
