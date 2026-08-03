"""Tier-credential storage: OS keyring, keyed by the full operating point
(`credential_key`: tier-model-effort-thinking), not by model alone — two
tiers bound to the same model still hold independent keys, and changing a
tier's effort/thinking stages a fresh key rather than silently reusing
whatever was stored for a different operating point. Pattern taken from
buddy-hub's `keyring` usage: a fixed service string, no fallback when the OS
has no backend (fails loud, matches house style — hardening deferred).
"""

from __future__ import annotations

import keyring
import keyring.errors

_SERVICE = "gekai"


def credential_key(tier: str, model: str, effort: str, thinking: bool) -> str:
    """The keyring account name for one tier's operating point."""
    return f"{tier}-{model}-{effort}-{'y' if thinking else 'n'}"


def has_api_key(tier_name: str) -> bool:
    return keyring.get_password(_SERVICE, tier_name) is not None


def get_api_key(tier_name: str) -> str:
    api_key = keyring.get_password(_SERVICE, tier_name)
    if api_key is None:
        raise LookupError(f"no API key stored for tier {tier_name!r}")
    return api_key


def set_api_key(tier_name: str, api_key: str) -> None:
    keyring.set_password(_SERVICE, tier_name, api_key)


def delete_api_key(tier_name: str) -> None:
    try:
        keyring.delete_password(_SERVICE, tier_name)
    except keyring.errors.PasswordDeleteError:
        pass  # already absent — deleting a non-existent credential is a no-op, not an error


__all__ = ["credential_key", "has_api_key", "get_api_key", "set_api_key", "delete_api_key"]
