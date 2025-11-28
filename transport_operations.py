from __future__ import annotations

import logging
import re
from typing import List

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
TIME_PATTERN = re.compile(
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?",
    re.IGNORECASE,
)


def _extract_weekdays(text: str) -> List[str]:
    text_lower = text.lower()
    return [day.title() for day in WEEKDAY_NAMES if day in text_lower]


def _extract_times(text: str) -> List[str]:
    times: List[str] = []
    for match in TIME_PATTERN.finditer(text):
        hour = match.group("hour")
        minute = match.group("minute") or "00"
        ampm = (match.group("ampm") or "").replace(".", "").upper()
        formatted = f"{hour}:{minute}"
        if ampm:
            formatted = f"{formatted} {ampm}"
        times.append(formatted.strip())
    return times


@traceable(name="hr_transport_placeholder")
def acknowledge_transport_request(user_message: str) -> OperationResult:
    """
    Dummy positive response for transport bookings.
    """
    times = _extract_times(user_message)
    weekdays = _extract_weekdays(user_message)

    pickup = times[0] if times else "the scheduled time"
    drop = times[1] if len(times) > 1 else None
    day_hint = weekdays[0] if weekdays else "the requested day"

    logger.info(
        "Transport placeholder: pickup=%s drop=%s day=%s raw=%s",
        pickup,
        drop,
        day_hint,
        user_message,
    )

    message = f"🚗 Cab booked for {day_hint} around {pickup}."
    if drop:
        message += f" Return ride is blocked near {drop}."
    message += " I'll push this to the transport desk once the integration is ready."

    metadata = {
        "pickup_time": pickup,
        "return_time": drop,
        "day_hint": day_hint,
    }
    return OperationResult(success=True, message=message, metadata=metadata)


__all__ = ["acknowledge_transport_request"]
