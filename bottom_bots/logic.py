"""Three-valued (Kleene) logic: True / False / None for unknown.

A checklist item is MET only when every input it needs is known and passes; one
known failure makes it NOT_MET; anything else is UNKNOWN. Missing data can never
add to the score.
"""

from __future__ import annotations

from .models import Status

Tri = bool | None


def all_of(*values: Tri) -> Tri:
    if any(v is False for v in values):
        return False
    if all(v is True for v in values):
        return True
    return None


def any_of(*values: Tri) -> Tri:
    if any(v is True for v in values):
        return True
    if all(v is False for v in values):
        return False
    return None


def status_of(value: Tri) -> Status:
    if value is True:
        return Status.MET
    if value is False:
        return Status.NOT_MET
    return Status.UNKNOWN
