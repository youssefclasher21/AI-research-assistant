from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    error: str | None = None
    text: str = ""


def validate_user_input(message: str, max_chars: int) -> ValidationResult:
    if message is None:
        return ValidationResult(ok=False, error="Please enter a research topic or question.")
    text = str(message).strip()
    if not text:
        return ValidationResult(ok=False, error="Please enter a research topic or question.")
    if len(text) > max_chars:
        return ValidationResult(
            ok=False,
            error=f"Message is too long (max {max_chars} characters).",
        )
    return ValidationResult(ok=True, text=text)
