import functools
import inspect
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any, ParamSpec

P = ParamSpec("P")
logger = logging.getLogger(__name__)


def log_llm_generate(
    fn: Callable[P, Coroutine[Any, Any, str]],
) -> Callable[P, Coroutine[Any, Any, str]]:
    """Log latency, prompt/response sizes, and model (if adapter exposes `_model`)."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> str:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        params = bound.arguments
        system_prompt = str(params.get("system_prompt", "") or "")
        user_prompt = str(params.get("user_prompt", "") or "")
        inst = params.get("self")
        model = getattr(inst, "_model", None) if inst is not None else None

        started = time.perf_counter()
        try:
            result = await fn(*args, **kwargs)
        except Exception:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "llm.generate failed model=%r duration_ms=%.2f system_chars=%d user_chars=%d",
                model,
                duration_ms,
                len(system_prompt),
                len(user_prompt),
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "llm.generate ok model=%r duration_ms=%.2f system_chars=%d user_chars=%d response_chars=%d",
            model,
            duration_ms,
            len(system_prompt),
            len(user_prompt),
            len(result),
        )
        return result

    return wrapper
