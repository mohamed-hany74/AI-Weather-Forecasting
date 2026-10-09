"""
Custom IP-based rate limiting decorator for Django views using cache backend.
Enforces 60 requests/minute per client IP and returns HTTP 429 Too Many Requests.
"""

import time
import functools
from django.core.cache import cache
from django.http import JsonResponse, HttpRequest

from .serializers import format_error_response


def get_client_ip(request: HttpRequest) -> str:
    """Extract client IP handling proxies safely."""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0].strip()
    else:
        ip = request.META.get('REMOTE_ADDR', '127.0.0.1')
    return ip


def ratelimit(requests_per_minute: int = 60):
    """
    Decorator that throttles view execution using a sliding-window in cache.
    Returns HTTP 429 when exceeded.
    """
    def decorator(view_func):
        @functools.wraps(view_func)
        def _wrapped_view(request: HttpRequest, *args, **kwargs):
            ip = get_client_ip(request)
            now = time.time()
            cache_key = f"ratelimit_{view_func.__name__}_{ip}"

            # Sliding window bucket
            history = cache.get(cache_key, [])
            # Purge timestamps older than 60 seconds
            history = [ts for ts in history if now - ts < 60.0]

            if len(history) >= requests_per_minute:
                retry_after = int(60.0 - (now - history[0])) if history else 60
                response = JsonResponse(
                    format_error_response(
                        f"Request rate limit exceeded. Max {requests_per_minute} requests per minute.",
                        code="RATE_LIMIT_EXCEEDED",
                        details={"retry_after_seconds": max(1, retry_after)}
                    ),
                    status=429
                )
                response["Retry-After"] = str(max(1, retry_after))
                return response

            history.append(now)
            cache.set(cache_key, history, timeout=65)

            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator
