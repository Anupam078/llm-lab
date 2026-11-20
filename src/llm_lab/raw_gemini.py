"""A small, direct HTTP client for Gemini's generateContent endpoint."""

import os
from typing import Any

import httpx
from dotenv import load_dotenv

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
MODEL = "gemini-3.5-flash"
TIMEOUT_SECONDS = 30.0


class LLMResponseError(ValueError):
    """Raised when Gemini returns a response without usable completed text."""


class GeminiAPIError(RuntimeError):
    """Raised when Gemini returns an unsuccessful HTTP status."""


def build_request(
    api_key: str,
    prompt: str,
    model: str = MODEL,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build the URL, headers, and JSON body without performing I/O.

    Spring analogy: this prepares the URL plus the HttpEntity inputs. Strictly,
    HttpEntity contains the headers and body; RestTemplate.exchange receives
    the URL separately.
    """
    url = f"{BASE_URL}/models/{model}:generateContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}],
            }
        ]
    }
    return url, headers, body


def extract_text(data: dict[str, Any]) -> str:
    """Map Gemini's response dictionary to application text, like DTO mapping.

    This function is pure: it only inspects the supplied dictionary and does
    not perform network, file, environment, or console I/O.
    """
    candidates = data.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise LLMResponseError("Gemini response contains no candidates.")

    first_candidate = candidates[0]
    if not isinstance(first_candidate, dict):
        raise LLMResponseError("Gemini's first candidate is not a valid object.")

    finish_reason = first_candidate.get("finishReason")
    if finish_reason != "STOP":
        raise LLMResponseError(
            "Gemini did not finish normally: "
            f"finishReason={finish_reason!r}; expected 'STOP'."
        )

    content = first_candidate.get("content")
    parts = content.get("parts") if isinstance(content, dict) else None

    text_parts: list[str] = []
    if isinstance(parts, list):
        for part in parts:
            if not isinstance(part, dict):
                continue
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text.strip():
                text_parts.append(part_text)

    if not text_parts:
        raise LLMResponseError(
            "Gemini's first candidate contains no non-empty text parts."
        )

    # A newline makes separate text parts readable while preserving their order.
    return "\n".join(text_parts)


def call_gemini(
    api_key: str,
    prompt: str,
    model: str = MODEL,
    timeout: float = TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Send the HTTP request and return its decoded JSON response.

    Spring analogy: httpx.post(...) is the counterpart to
    RestTemplate.exchange(...). The HTTP status is checked before JSON parsing.
    """
    url, headers, body = build_request(api_key, prompt, model)
    response = httpx.post(url, headers=headers, json=body, timeout=timeout)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Include useful provider diagnostics, but never echo the API key.
        error_body = exc.response.text or "<empty response body>"
        if api_key:
            error_body = error_body.replace(api_key, "[REDACTED]")
        raise GeminiAPIError(
            f"Gemini API returned HTTP {exc.response.status_code}: {error_body}"
        ) from None

    return response.json()


def main() -> None:
    """Application entry point: load configuration, exchange, then map the DTO."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file."
        )

    data = call_gemini(api_key, "Explain HTTP in one sentence.")
    print(extract_text(data))

    usage = data.get("usageMetadata")
    if usage is not None:
        print("Usage metadata:", usage)


if __name__ == "__main__":
    main()
