from __future__ import annotations

from typing import Tuple
import re


# ---------------------------------------------------------------------
# BANNED WORD LISTS
# You can expand these as needed.
# ---------------------------------------------------------------------

TOXIC_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard",
    "dick", "cunt", "moron", "retard",
]

RACIST_SLURS = [
    "nigger", "nigga", "chink", "spic", "paki",
    "kike", "faggot", "tranny",
]

PROMPT_INJECTION = [
    "ignore all previous instructions",
    "disregard previous instructions",
    "jailbreak",
    "system override",
    "act as an unrestricted ai",
]


def _find_banned_words(text: str) -> list[str]:
    """
    Return a list of banned words actually present in the text (case-insensitive).
    Uses word-boundary matching to avoid partial matches.
    """
    lower = text.lower()
    found: list[str] = []
    for w in TOXIC_WORDS + RACIST_SLURS:
        if re.search(rf"\b{re.escape(w)}\b", lower, flags=re.IGNORECASE):
            found.append(w)
    return found


def _highlight_banned_in_text(text: str, banned: list[str]) -> str:
    """
    Return the original text with all banned words replaced by strikethrough
    Markdown, e.g. 'badword' -> '~~badword~~', case-insensitive.
    """
    highlighted = text
    for w in set(banned):
        pattern = rf"\b{re.escape(w)}\b"

        def replacer(match: re.Match) -> str:
            original = match.group(0)
            return f"~~{original}~~"

        highlighted = re.sub(pattern, replacer, highlighted, flags=re.IGNORECASE)
    return highlighted


# ---------------------------------------------------------------------
# VALIDATE USER INPUT (block unsafe, highlight offending words)
# ---------------------------------------------------------------------

def validate_input(message: str) -> Tuple[bool, str | None]:
    """
    Input guard rails:
      - empty / max length checks
      - toxic language detection
      - racist language detection
      - prompt injection detection

    If invalid (False):
      - returns a message like:
          "Your message violates company communication policy ..."
        and shows the original text with offending words struck through.

    Returns:
        (is_allowed, cleaned_or_error_message)
    """
    text = message.strip()

    if not text:
        return False, "Please type something to continue."

    # Limit length (basic DOS guard)
    if len(text) > 2000:
        return False, "Your message is too long. Please shorten and try again."

    lower = text.lower()

    # Simple prompt injection block
    if any(p in lower for p in PROMPT_INJECTION):
        return (
            False,
            "Your message violates company communication policy: "
            "system override / jailbreak instructions are not allowed.",
        )

    # Toxic & racist filtering
    banned_found = _find_banned_words(text)
    if banned_found:
        # Highlight offending words inside the original message
        highlighted = _highlight_banned_in_text(text, banned_found)

        unique_words = sorted(set(banned_found))
        banned_display = ", ".join(f"~~{w}~~" for w in unique_words)

        response = (
            "Your message violates company communication policy because it "
            f"contains restricted word(s): {banned_display}.\n\n"
            "**Original message with problematic words highlighted:**\n\n"
            f"> {highlighted}"
        )

        return False, response

    # If we reach here, input is allowed
    return True, text


# ---------------------------------------------------------------------
# VALIDATE OUTPUT (soft clean)
# ---------------------------------------------------------------------

def validate_output(answer: str) -> str:
    """
    Output guard rails:
      - Remove generic LLM disclaimers
      - Mask any banned words in the assistant response

    This is a soft cleanup layer and does not block the response.
    """
    text = answer.strip()
    lower = text.lower()

    # Remove disclaimers like "As an AI language model..."
    if "as an ai" in lower or "as a language model" in lower:
        cleaned_lines = [
            ln
            for ln in text.splitlines()
            if "as an ai" not in ln.lower()
            and "as a language model" not in ln.lower()
        ]
        text = "\n".join(cleaned_lines).strip()
        lower = text.lower()

    # Mask banned words in output with ***
    for w in set(TOXIC_WORDS + RACIST_SLURS):
        text = re.sub(
            rf"\b{re.escape(w)}\b",
            "***",
            text,
            flags=re.IGNORECASE,
        )

    return text
