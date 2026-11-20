import httpx
import pytest

from llm_lab import raw_gemini


def test_extract_text_from_one_text_part() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [{"text": "HTTP is a request-response protocol."}],
                },
            }
        ]
    }

    assert (
        raw_gemini.extract_text(response)
        == "HTTP is a request-response protocol."
    )


def test_extract_text_joins_all_text_parts_in_order() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {
                    "parts": [
                        {"text": "First part."},
                        {"text": "Second part."},
                    ]
                },
            }
        ]
    }

    assert raw_gemini.extract_text(response) == "First part.\nSecond part."


def test_extract_text_rejects_empty_candidates() -> None:
    with pytest.raises(raw_gemini.LLMResponseError, match="no candidates"):
        raw_gemini.extract_text({"candidates": []})


def test_extract_text_rejects_non_stop_finish_reason() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": "Partial answer"}]},
            }
        ]
    }

    with pytest.raises(raw_gemini.LLMResponseError, match="MAX_TOKENS"):
        raw_gemini.extract_text(response)


def test_extract_text_rejects_missing_text() -> None:
    response = {
        "candidates": [
            {
                "finishReason": "STOP",
                "content": {"parts": [{"inline_data": "not text"}]},
            }
        ]
    }

    with pytest.raises(raw_gemini.LLMResponseError, match="no non-empty text"):
        raw_gemini.extract_text(response)


def test_build_request_puts_model_in_url_and_key_in_headers() -> None:
    api_key = "test-secret-key"
    model = "test-model"

    url, headers, body = raw_gemini.build_request(
        api_key=api_key,
        prompt="Explain HTTP.",
        model=model,
    )

    assert url.endswith(f"/models/{model}:generateContent")
    assert headers["x-goog-api-key"] == api_key
    assert headers["Content-Type"] == "application/json"
    assert body["contents"][0]["parts"][0]["text"] == "Explain HTTP."
    assert "model" not in body


def test_call_gemini_surfaces_http_error_body_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api_key = "test-secret-key"
    request = httpx.Request(
        "POST",
        "https://generativelanguage.googleapis.com/v1beta/models/test-model:generateContent",
    )
    response = httpx.Response(
        status_code=400,
        text=f'{{"message":"Invalid request; leaked value: {api_key}"}}',
        request=request,
    )

    def fake_post(*args: object, **kwargs: object) -> httpx.Response:
        return response

    monkeypatch.setattr(raw_gemini.httpx, "post", fake_post)

    with pytest.raises(raw_gemini.GeminiAPIError) as error:
        raw_gemini.call_gemini(api_key, "Hello", model="test-model")

    message = str(error.value)
    assert "HTTP 400" in message
    assert "Invalid request" in message
    assert "[REDACTED]" in message
    assert api_key not in message
