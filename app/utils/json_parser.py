"""
Utility helpers for safe JSON parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any


_JSON_FENCE_PATTERN = re.compile(
    r"```(?:json)?\s*([\s\S]*?)```",
    re.IGNORECASE,
)


def extract_json_block(text: str) -> str:
    """
    Extract JSON content from a model response.

    If the response contains a Markdown code fence such as:

        ```json
        {"key": "value"}
        ```

    only the content inside the fence is returned.

    If no code fence is present, the complete response is returned.
    """

    if not isinstance(text, str):
        raise ValueError(
            "Expected model response to be a string."
        )

    cleaned = text.strip()

    match = _JSON_FENCE_PATTERN.search(cleaned)

    if match:
        return match.group(1).strip()

    return cleaned


def safe_parse_json(text: str) -> dict[str, Any]:
    """
    Parse a JSON object from a model response.

    Markdown JSON code fences are supported.

    Raises:
        ValueError:
            If the response is not valid JSON or does not contain
            a JSON object.
    """

    raw = extract_json_block(text)

    if not raw:
        raise ValueError(
            "Model response contained no JSON content."
        )

    try:
        parsed = json.loads(raw)

    except json.JSONDecodeError as exc:
        preview = text[:500]

        raise ValueError(
            "Could not parse JSON from model response: "
            f"{exc}\n\nRaw text:\n{preview}"
        ) from exc

    if not isinstance(parsed, dict):
        raise ValueError(
            "Expected the model response to contain a JSON object."
        )

    return parsed