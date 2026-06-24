"""Input validation: length limits and prompt injection detection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Patterns that commonly indicate prompt injection attempts
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+previous\s+(instructions?|commands?)", re.IGNORECASE),
    re.compile(r"disregard\s+above\s+(instructions?|rules?)", re.IGNORECASE),
    re.compile(r"system\s*:\s*(prompt|instruction)", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[?\s*developer\s*(mode|message)\s*\]?", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if)?\s*a\s+(new|different)", re.IGNORECASE),
    re.compile(r"pretend\s+to\s+be\s+a\s+(new|different)", re.IGNORECASE),
    re.compile(r"you\s+are\s+(now\s+)?(a\s+)?(?:new|different)\s+", re.IGNORECASE),
]


class InputValidator:
    """Validate user input for security and policy compliance."""

    def __init__(
        self,
        *,
        max_length: int = 4096,
        injection_detection: bool = True,
    ) -> None:
        self._max_length = max_length
        self._injection_detection = injection_detection

    def validate(self, text: str) -> ValidationResult:
        """Validate input text. Returns result with errors/warnings."""
        errors: list[str] = []
        warnings: list[str] = []

        # Length check
        if len(text) > self._max_length:
            errors.append(
                f"Input too long: {len(text)} chars (max {self._max_length})"
            )

        # Prompt injection detection
        if self._injection_detection:
            for pattern in _INJECTION_PATTERNS:
                match = pattern.search(text)
                if match:
                    warnings.append(
                        f"Suspicious pattern detected: '{match.group()}'"
                    )
                    logger.warning(
                        "Potential prompt injection: pattern=%s text_preview=%s",
                        pattern.pattern,
                        text[:100],
                    )

        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )

    def validate_or_raise(self, text: str, block_injections: bool = False) -> None:
        """Validate and raise ValueError if invalid.

        Args:
            text: Input to validate.
            block_injections: If True, treat injection warnings as errors.
        """
        result = self.validate(text)
        if not result.is_valid:
            raise ValueError(f"Input validation failed: {'; '.join(result.errors)}")
        if block_injections and result.warnings:
            raise ValueError(
                f"Input blocked due to suspicious patterns: {'; '.join(result.warnings)}"
            )
