"""Keep the search client available to existing UI callers."""

from ..serving.client import DEFAULT_API_URL, SearchError, search

__all__ = ["DEFAULT_API_URL", "SearchError", "search"]
