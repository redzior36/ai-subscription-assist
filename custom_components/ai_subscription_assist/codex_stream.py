"""Read a complete Codex Responses SSE stream before running any tools."""

from collections.abc import AsyncIterable, AsyncIterator
import json
from typing import Any

import httpx
from homeassistant.exceptions import HomeAssistantError

from .provider_errors import (
    extract_provider_error_detail,
    format_http_error_message,
    format_rate_limited_message,
)


async def _sse_events(lines: AsyncIterable[str]) -> AsyncIterator[dict[str, Any]]:
    data: list[str] = []
    async for line in lines:
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line and data:
            raw = "\n".join(data)
            data.clear()
            if raw == "[DONE]":
                return
            try:
                event = json.loads(raw)
            except ValueError as err:
                raise HomeAssistantError("OpenAI Codex sent invalid SSE data") from err
            if not isinstance(event, dict):
                raise HomeAssistantError("OpenAI Codex sent an invalid SSE event")
            yield event


async def async_codex_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Return the response from the terminal event, with item events as fallback."""
    try:
        async with client.stream(
            "POST", url, headers=headers, json=payload, timeout=60
        ) as resp:
            if not resp.is_success:
                detail = extract_provider_error_detail(
                    await resp.aread(), content_type=resp.headers.get("content-type")
                )
                if resp.status_code == 429:
                    raise HomeAssistantError(
                        format_rate_limited_message(
                            "OpenAI Codex",
                            retry_after=resp.headers.get("retry-after"),
                            detail=detail,
                        )
                    )
                message = format_http_error_message("OpenAI Codex", resp.status_code)
                if detail and resp.status_code not in (401, 403):
                    message = f"{message} Provider message: {detail}"
                raise HomeAssistantError(message)

            items: list[dict[str, Any]] = []
            async for event in _sse_events(resp.aiter_lines()):
                kind = event.get("type")
                if kind == "response.output_item.done" and isinstance(
                    event.get("item"), dict
                ):
                    items.append(event["item"])
                elif kind in ("response.completed", "response.done"):
                    response = event.get("response")
                    if not isinstance(response, dict):
                        raise HomeAssistantError(
                            "OpenAI Codex sent an invalid response"
                        )
                    if response.get("status") not in (None, "completed"):
                        raise HomeAssistantError(
                            "OpenAI Codex did not complete the response"
                        )
                    return {"output": items, **response}
                elif kind in ("error", "response.failed", "response.incomplete"):
                    response = event.get("response")
                    error = event.get("error") or event
                    if isinstance(response, dict):
                        error = (
                            response.get("error")
                            or response.get("incomplete_details")
                            or error
                        )
                    detail = extract_provider_error_detail(json.dumps(error))
                    raise HomeAssistantError(
                        f"OpenAI Codex stream failed: {detail or kind}"
                    )
    except httpx.HTTPError as err:
        raise HomeAssistantError(
            f"OpenAI Codex connection failed: {type(err).__name__}"
        ) from err

    raise HomeAssistantError("OpenAI Codex stream ended before the response completed")
