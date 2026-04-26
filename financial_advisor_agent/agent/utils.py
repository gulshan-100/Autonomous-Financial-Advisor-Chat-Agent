# -*- coding: utf-8 -*-
"""
Shared utilities for the Financial Advisor Agent.

sanitize_for_llm() is the critical safety function called before every
LLM API call to strip surrogate characters that cause UTF-8 encoding errors.

Root cause: emoji typed in the browser (e.g. chart emoji, U+1F4CA) can
arrive from JS as surrogate pairs (\\uD83D\\uDCCA). Python's str allows
these surrogates internally but json.dumps and HTTP UTF-8 serialisation
both reject them with 'surrogates not allowed'.
"""
from __future__ import annotations

import json
import re


def sanitize_for_llm(text: str | None) -> str:
    """
    Strip surrogate characters and non-encodable codepoints from a string
    before passing it to the OpenAI API.

    Strategy:
      1. Encode to UTF-8 with errors='replace'  -> surrogates become b'?'
      2. Decode back to str                     -> clean unicode
      3. Collapse multiple '??' into single '?'  -> cosmetic
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Replace surrogates during encode, then decode clean
    cleaned = text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    # Collapse repeated replacement markers (optional cosmetic step)
    cleaned = re.sub(r"\?{2,}", "?", cleaned)
    return cleaned


def sanitize_dict(obj: dict | list | str | None) -> dict | list | str:
    """
    Recursively sanitize all string values inside a dict or list.
    Used to clean data that will be JSON-serialised into LLM prompts.
    """
    if isinstance(obj, dict):
        return {k: sanitize_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_dict(item) for item in obj]
    if isinstance(obj, str):
        return sanitize_for_llm(obj)
    return obj


def safe_json_dumps(obj, **kwargs) -> str:
    """
    JSON-serialise an object after sanitizing all string values.
    Prevents surrogate-related errors in prompt building.

    Uses ensure_ascii=True by default so all non-ASCII chars are
    \\uXXXX-escaped, which is always safe for the OpenAI API.
    """
    # Remove any caller-supplied ensure_ascii so we always control it
    kwargs.pop("ensure_ascii", None)
    return json.dumps(sanitize_dict(obj), ensure_ascii=True, **kwargs)
