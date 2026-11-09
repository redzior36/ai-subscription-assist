"""Home Assistant AI Task support for ChatGPT subscriptions."""

from __future__ import annotations

import base64
import io
import json
from uuid import uuid4

import httpx
from PIL import Image
import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import AiSubscriptionAssistConfigEntry, OpenAICodexClient
from .const import CONF_CODEX_IMAGE_GENERATION, SUBENTRY_AI_TASK
from .entity import AiSubscriptionAssistBaseLLMEntity, _codex_image_parts
from .provider_errors import extract_provider_error_detail, format_rate_limited_message

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: AiSubscriptionAssistConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Register each configured AI Task provider with Home Assistant."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type == SUBENTRY_AI_TASK:
            async_add_entities(
                [CodexAITaskEntity(config_entry, subentry)],
                config_subentry_id=subentry.subentry_id,
            )


class CodexAITaskEntity(ai_task.AITaskEntity, AiSubscriptionAssistBaseLLMEntity):
    """Generate data through the configured ChatGPT subscription."""

    def __init__(
        self, entry: AiSubscriptionAssistConfigEntry, subentry: ConfigSubentry
    ) -> None:
        """Advertise image generation only when enabled for this entity."""
        super().__init__(entry, subentry)
        self._attr_supported_features = (
            ai_task.AITaskEntityFeature.GENERATE_DATA
            | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
        )
        if subentry.data.get(CONF_CODEX_IMAGE_GENERATION, False):
            self._attr_supported_features |= ai_task.AITaskEntityFeature.GENERATE_IMAGE

    async def _async_generate_data(
        self, task: ai_task.GenDataTask, chat_log: conversation.ChatLog
    ) -> ai_task.GenDataTaskResult:
        """Return text or data validated against the task's schema."""
        await self._async_handle_chat_log_openai_codex(
            chat_log, "task_result", task.structure
        )
        response = chat_log.content[-1]
        if (
            not isinstance(response, conversation.AssistantContent)
            or not response.content
        ):
            raise HomeAssistantError("ChatGPT did not return an AI Task result")

        data = response.content
        if task.structure is not None:
            try:
                data = task.structure(json.loads(data))
            except (json.JSONDecodeError, vol.Invalid) as err:
                raise HomeAssistantError(
                    "ChatGPT returned an AI Task result that does not match the requested structure"
                ) from err
        return ai_task.GenDataTaskResult(chat_log.conversation_id, data)

    async def _async_generate_image(
        self, task: ai_task.GenImageTask, chat_log: conversation.ChatLog
    ) -> ai_task.GenImageTaskResult:
        """Call the native Codex image service and return the image to Home Assistant."""
        client = self.entry.runtime_data
        if not isinstance(client, OpenAICodexClient):
            raise HomeAssistantError("The ChatGPT Codex client is unavailable")

        operation = "generations"
        body: dict[str, object] = {
            "model": "gpt-image-2",
            "prompt": task.instructions,
            "n": 1,
        }
        if task.attachments:
            parts = await self.hass.async_add_executor_job(
                _codex_image_parts, task.attachments
            )
            body["images"] = [{"image_url": part["image_url"]} for part in parts]
            operation = "edits"

        base_url = (
            client.base_url.rstrip("/")
            .removesuffix("/responses")
            .removesuffix("/codex")
        )
        headers = {
            **client._headers(),
            "originator": "ai_subscription_assist",
            "x-codex-image-turn-id": uuid4().hex,
        }
        try:
            response = await client.http_client.post(
                f"{base_url}/codex/images/{operation}",
                headers=headers,
                json=body,
                timeout=httpx.Timeout(240.0, connect=10.0),
                follow_redirects=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as err:
            response = err.response
            detail = extract_provider_error_detail(
                response.content, content_type=response.headers.get("content-type")
            )
            if response.status_code == 429:
                raise HomeAssistantError(
                    format_rate_limited_message(
                        "ChatGPT image generation",
                        retry_after=response.headers.get("retry-after"),
                        detail=detail,
                    )
                ) from err
            message = f"ChatGPT image generation returned HTTP {response.status_code}"
            if detail:
                message = f"{message}: {detail}"
            raise HomeAssistantError(message) from err
        except httpx.HTTPError as err:
            raise HomeAssistantError(
                f"Could not reach the ChatGPT image service ({type(err).__name__})"
            ) from err

        return await self.hass.async_add_executor_job(
            _decode_image_result, response.content, chat_log.conversation_id
        )


def _decode_image_result(
    body: bytes, conversation_id: str
) -> ai_task.GenImageTaskResult:
    """Decode and inspect the generated image outside the event loop."""
    try:
        item = json.loads(body)["data"][0]
        image_data = base64.b64decode(item["b64_json"], validate=True)
        with Image.open(io.BytesIO(image_data)) as image:
            mime_type = Image.MIME.get(image.format)
            width, height = image.size
            image.verify()
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        OSError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as err:
        raise HomeAssistantError("ChatGPT returned an invalid image") from err
    if mime_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HomeAssistantError("ChatGPT returned an unsupported image format")
    return ai_task.GenImageTaskResult(
        image_data=image_data,
        conversation_id=conversation_id,
        mime_type=mime_type,
        width=width,
        height=height,
    )
