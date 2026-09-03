"""Model-credential storage: OS keyring, keyed by `provider:model` — a
property of the model, not of whatever tier happens to be bound to it. A key
is entered once via `/models` and stays valid however the tiers are later
reshuffled by `/tier`. The provider prefix namespaces the account name so a
future second wire protocol offering the same model name doesn't collide.
Pattern taken from buddy-hub's `keyring` usage: a fixed service string, no
fallback when the OS has no backend (fails loud, matches house style —
hardening deferred).
"""

from __future__ import annotations

import keyring
import keyring.errors

_SERVICE = "gekai"


def credential_key(provider: str, model: str) -> str:
    """The keyring account name for one model's API key."""
    return f"{provider}:{model}"


def has_api_key(cred_key: str) -> bool:
    return keyring.get_password(_SERVICE, cred_key) is not None


def get_api_key(cred_key: str) -> str:
    api_key = keyring.get_password(_SERVICE, cred_key)
    if api_key is None:
        raise LookupError(f"no API key stored for {cred_key!r}")
    return api_key


def set_api_key(cred_key: str, api_key: str) -> None:
    keyring.set_password(_SERVICE, cred_key, api_key)


def delete_api_key(cred_key: str) -> None:
    try:
        keyring.delete_password(_SERVICE, cred_key)
    except keyring.errors.PasswordDeleteError:
        pass  # already absent — deleting a non-existent credential is a no-op, not an error


__all__ = ["credential_key", "has_api_key", "get_api_key", "set_api_key", "delete_api_key"]
