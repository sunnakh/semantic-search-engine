"""Resolve the running API's active model for candidate evaluation."""

import json
import re
from dataclasses import fields
from urllib.error import URLError
from urllib.request import urlopen

from ..serving.release import Release


def served_release(api_url: str) -> Release:
    try:
        with urlopen(api_url.rstrip("/") + "/api/v1/models", timeout=5) as response:
            payload = response.read(65537)
        if len(payload) > 65536:
            raise ValueError("serving metadata exceeds 64 KiB")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("serving metadata must be an object")
        metadata = Release(
            **{field.name: value[field.name] for field in fields(Release)}
        )
        if (
            any(
                not isinstance(item, str)
                for name, item in vars(metadata).items()
                if name != "dense_timeout_ms"
            )
            or not re.fullmatch(r"[1-9][0-9]*", metadata.model_version)
            or type(metadata.dense_timeout_ms) is not int
            or metadata.dense_timeout_ms < 1
        ):
            raise ValueError("invalid served release metadata")
        return metadata
    except (URLError, OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError(
            f"cannot resolve the active serving release: {error}"
        ) from error
