from __future__ import annotations

from typing import Any, Callable

try:
    from langsmith import traceable as _ls_traceable
except ImportError:  # pragma: no cover

    def _ls_traceable(*args: Any, **kwargs: Any) -> Callable:
        def decorator(fn: Callable) -> Callable:
            return fn

        return decorator


traceable = _ls_traceable

