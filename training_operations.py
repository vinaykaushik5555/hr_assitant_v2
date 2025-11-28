from __future__ import annotations

import logging
from typing import List

from langsmith import traceable

from hr_operation_result import OperationResult


logger = logging.getLogger(__name__)


def _mock_pending_trainings() -> List[str]:
    return [
        "Security Awareness 2025",
        "Expense Compliance Basics",
    ]


def _mock_overdue_trainings() -> List[str]:
    return [
        "Code of Conduct Refresher",
    ]


@traceable(name="hr_training_placeholder")
def acknowledge_training_status(user_message: str) -> OperationResult:
    """
    Dummy positive response for training status queries.
    """
    pending = _mock_pending_trainings()
    overdue = _mock_overdue_trainings()

    logger.info(
        "Training placeholder: pending=%s overdue=%s raw=%s",
        pending,
        overdue,
        user_message,
    )

    lines = ["📚 Training summary:"]
    lines.append("- Pending: " + ", ".join(pending))
    if overdue:
        lines.append("- Overdue: " + ", ".join(overdue))
    else:
        lines.append("- Overdue: None 🎉")
    lines.append("I'll sync these with the LMS once the integration is wired.")

    metadata = {
        "pending_trainings": pending,
        "overdue_trainings": overdue,
    }
    return OperationResult(success=True, message="\n".join(lines), metadata=metadata)


__all__ = ["acknowledge_training_status"]
