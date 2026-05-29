"""Langfuse callback wiring (langfuse 4.x).

In langfuse 4.x the LangChain callback handler lives at
``langfuse.langchain.CallbackHandler``. It reads credentials from the
environment (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST), so we
only return a handler when those keys are present; otherwise tracing is a no-op.
"""

from __future__ import annotations

import os
from typing import Any


def get_callbacks() -> list[Any]:
    """Return LangChain callbacks: ``[]`` when Langfuse keys are absent."""
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]
