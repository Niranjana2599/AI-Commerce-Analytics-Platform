"""Small response-side PII redaction helper for common contact details."""

import re

EMAIL_PATTERN=re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
# Require a phone-like run of 10–15 digits and keep UUID hyphens/alphanumeric
# boundaries out of matches so ordinary commerce identifiers survive redaction.
PHONE_PATTERN=re.compile(r"(?<![\w-])\+?(?:\d[ .()\-]*){9,14}\d(?![\w-])")


def contains_contact_details(text:str)->bool:
    return bool(EMAIL_PATTERN.search(text) or PHONE_PATTERN.search(text))


def redact_contact_details(text: str) -> str:
    text=EMAIL_PATTERN.sub("[REDACTED_EMAIL]",text)
    return PHONE_PATTERN.sub("[REDACTED_PHONE]",text)
