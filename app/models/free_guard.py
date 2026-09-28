"""Free-tier enforcement for model ids.

The rule is that every LLM call in this project goes through OpenRouter on a
free-tier model, so an id that is neither `openrouter/free` nor a `:free`
variant is a configuration error. `app.config` refuses it at startup and
`app.models.llm.get_llm` refuses it again on every call, in case settings were
built by hand.
"""

from __future__ import annotations

from ..config import FREE_MODEL_IDS, is_free_model


class NonFreeModelError(ValueError):
    """Raised when a model id is not an OpenRouter free-tier model."""


def assert_free_model(model_id: str) -> str:
    """Return `model_id` if it is free-tier, otherwise raise.

    Raises:
        NonFreeModelError: with a message naming the offending id and the rule.
    """
    candidate = (model_id or "").strip()
    if not candidate:
        raise NonFreeModelError(
            "No model id given. Set LLM_MODELS to a free OpenRouter model: an id ending "
            f"in ':free' or one of {sorted(FREE_MODEL_IDS)}."
        )
    if not is_free_model(candidate):
        raise NonFreeModelError(
            f"Model {candidate!r} is not a free OpenRouter model. This project is only "
            f"allowed to use ids ending in ':free' or {sorted(FREE_MODEL_IDS)}; no paid "
            "LLM API may be used or required."
        )
    return candidate


def free_model_candidates(models: list[str]) -> list[str]:
    """Filter a model list down to the free ones, preserving order.

    Used by the fallback chain: a list that already passed settings validation is
    returned unchanged, and one built by hand still cannot smuggle in a paid model.
    """
    return [m for m in (model.strip() for model in models) if m and is_free_model(m)]
