from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class OperationResult:
    """
    Simple container for placeholder HR operations.
    """

    success: bool
    message: str
    metadata: Dict[str, object] = field(default_factory=dict)


__all__ = ["OperationResult"]
