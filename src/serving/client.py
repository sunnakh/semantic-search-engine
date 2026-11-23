"""HTTP client for the SciFact search service."""

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .schemas import SEARCH_PATH, SENTINEL_HEADER, SearchRequest, SearchResponse

DEFAULT_API_URL = "http://127.0.0.1:8000"


class SearchError(RuntimeError):
    """A failed search with a message suitable for the UI."""


def search(
    api_url: str, payload: SearchRequest, *, sentinel_token: str | None = None
) -> SearchResponse:
    headers = {"Content-Type": "application/json"}
    if sentinel_token is not None:
        headers[SENTINEL_HEADER] = sentinel_token
    try:
        request = Request(
            api_url.rstrip("/") + SEARCH_PATH,
            data=payload.model_dump_json().encode(),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return SearchResponse.model_validate_json(response.read())
    except HTTPError as error:
        error.close()
        raise SearchError(f"The search API returned HTTP {error.code}.") from error
    except (URLError, OSError) as error:
        raise SearchError(
            "The search API is unavailable or timed out. Try again."
        ) from error
    except ValidationError as error:
        raise SearchError("The search API returned an invalid response.") from error
    except ValueError as error:
        raise SearchError("Check the search API URL configuration.") from error
