from __future__ import annotations

import logging
import re
from typing import List, Optional

from langsmith import traceable

from hr_operation_result import OperationResult


logger = logging.getLogger(__name__)

WEEKDAY_NAMES = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]
HOUR_PATTERN = re.compile(r"(\d{1,2})\s*(?:hr|hrs|hour|hours)", re.IGNORECASE)
PROJECT_CODE_PATTERN = re.compile(r"\b[A-Z]{2,}-?\d+\b")


def _extract_weekdays(text: str) -> List[str]:
    text_lower = text.lower()
    return [day.title() for day in WEEKDAY_NAMES if day in text_lower]


def _extract_hours(text: str) -> Optional[int]:
    match = HOUR_PATTERN.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _extract_project_code(text: str) -> Optional[str]:
    match = PROJECT_CODE_PATTERN.search(text.upper())
    if match:
        return match.group(0)
    return None


@traceable(name="hr_timesheet_placeholder")
def acknowledge_timesheet_request(user_message: str) -> OperationResult:
    """
    Dummy positive response for timesheet capture.
    """
    hours = _extract_hours(user_message) or 8
    weekdays = _extract_weekdays(user_message) or [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
    ]
    project_code = _extract_project_code(user_message) or "PROJECT-CODE"

    logger.info(
        "Timesheet placeholder: hours=%s weekdays=%s project=%s raw=%s",
        hours,
        weekdays,
        project_code,
        user_message,
    )

    message = (
        f"✅ Logged {hours} hours per day for {', '.join(weekdays)} "
        f"against project {project_code}. I'll sync these with the real "
        "timesheet tool once it's connected."
    )
    metadata = {
        "hours_per_day": hours,
        "weekdays": weekdays,
        "project_code": project_code,
    }
    return OperationResult(success=True, message=message, metadata=metadata)


__all__ = ["acknowledge_timesheet_request"]
