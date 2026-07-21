"""Small, safe wrapper around the FastAPI backend."""

from typing import Any

import requests

from config import API_BASE_URL


def api_request(method: str, path: str, *, json: dict[str, Any] | None = None) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    """Call an API endpoint and return either JSON data or a user-friendly error."""
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", json=json, timeout=30)
        response.raise_for_status()
        return response.json(), None
    except requests.Timeout:
        return None, "The API request timed out. Please try again."
    except requests.ConnectionError:
        return None, f"Cannot reach the API at {API_BASE_URL}. Start the FastAPI backend first."
    except requests.HTTPError:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"API error ({response.status_code}): {detail}"
    except requests.RequestException as error:
        return None, f"Request failed: {error}"
