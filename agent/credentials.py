"""Model-credential storage (plan 28 Phase 1a): OS keyring, keyed by model
name, not by tier — one stored secret per model, reused across every tier
bound to it. Pattern taken from buddy-hub's `keyring` usage: a fixed service
string, no fallback when the OS has no backend (fails loud, matches house
style — hardening deferred).
"""

from __future__ import annotations

import keyring
import keyring.errors

_SERVICE = "gekai"


def has_api_key(model_name: str) -> bool:
    return keyring.get_password(_SERVICE, model_name) is not None


def get_api_key(model_name: str) -> str:
    api_key = keyring.get_password(_SERVICE, model_name)
    if api_key is None:
        raise LookupError(f"no API key stored for model {model_name!r}")
    return api_key


def set_api_key(model_name: str, api_key: str) -> None:
    keyring.set_password(_SERVICE, model_name, api_key)


def delete_api_key(model_name: str) -> None:
    try:
        keyring.delete_password(_SERVICE, model_name)
    except keyring.errors.PasswordDeleteError:
        pass  # already absent — deleting a non-existent credential is a no-op, not an error


__all__ = ["has_api_key", "get_api_key", "set_api_key", "delete_api_key"]
