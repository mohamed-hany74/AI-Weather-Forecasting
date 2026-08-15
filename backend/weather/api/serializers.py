"""
Serializers and response formatters for WeatherCast API.
"""

from typing import Dict, Any


def format_error_response(message: str, code: str = "ERROR", details: Any = None) -> Dict[str, Any]:
    """Format consistent, secure error responses."""
    resp = {
        "status": "error",
        "code": code,
        "message": message
    }
    if details:
        resp["details"] = details
    return resp
