"""Exercise ChatGPT AI Task responses with Home Assistant task and schema types."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("homeassistant")

import httpx
from PIL import Image
from homeassistant.components import ai_task, conversation
from homeassistant.components.ai_task.services import _validate_structure_fields
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError

from custom_components.ai_subscription_assist import OpenAICodexClient
from custom_components.ai_subscription_assist.ai_task import CodexAITaskEntity
from custom_components.ai_subscription_assist.config_flow import (
    AiSubscriptionAssistConfigFlow,
    CodexAITaskSubentryFlow,
)
from custom_components.ai_subscription_assist.const import (
    CONF_CHAT_MODEL,
    CONF_CODEX_IMAGE_GENERATION,
    CONF_PROVIDER,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CODEX,
    SUBENTRY_AI_TASK,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2sAAAAASUVORK5CYII="
)


def test_ai_task_is_available_for_codex_entries():
    for provider in (PROVIDER_OPENAI_CODEX, PROVIDER_OPENAI):
        entry = SimpleNamespace(data={CONF_PROVIDER: provider})
        subentries = AiSubscriptionAssistConfigFlow.async_get_supported_subentry_types(
            entry
        )
        if provider == PROVIDER_OPENAI_CODEX:
            assert subentries[SUBENTRY_AI_TASK] is CodexAITaskSubentryFlow
        else:
            assert SUBENTRY_AI_TASK not in subentries


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (None, "Widzę lampę."),
        ('{"objects":1}', {"objects": 1.0}),
        ('{"unrequested":1}', None),
    ],
)
def test_generate_codex_task_with_image(tmp_path, arguments, expected):
    structure = None
    if arguments is not None:
        structure = _validate_structure_fields(
            {"objects": {"required": True, "selector": {"number": {}}}}
        )
    image = tmp_path / "test.png"
    image.write_bytes(PNG)
    attachment = conversation.Attachment(
        "media-source://test/image", "image/png", image
    )

    def respond(request):
        payload = json.loads(request.content)
        assert request.headers["authorization"] == "Bearer test-token"
        assert payload["model"] == "gpt-6.1-terra"
        assert payload["input"][0]["content"] == [
            {"type": "input_text", "text": "Opisz zdjęcie."},
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64.b64encode(PNG).decode()}",
            },
        ]
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Widzę lampę."}],
            }
        ]
        if arguments is not None:
            assert payload["tools"][0]["name"] == "task_result"
            assert payload["tools"][0]["parameters"]["required"] == ["objects"]
            assert (
                payload["tools"][0]["parameters"]["properties"]["objects"]["type"]
                == "number"
            )
            output.append(
                {
                    "type": "function_call",
                    "name": "task_result",
                    "call_id": "call_result",
                    "arguments": arguments,
                }
            )
        event = {"type": "response.completed", "response": {"output": output}}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\n")

    async def run():
        hass = HomeAssistant(str(tmp_path))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            codex = OpenAICodexClient(
                "test-token", "test-account", "https://example.test", client
            )
            entity = CodexAITaskEntity(
                SimpleNamespace(
                    data={CONF_PROVIDER: PROVIDER_OPENAI_CODEX}, runtime_data=codex
                ),
                SimpleNamespace(
                    data={CONF_CHAT_MODEL: "gpt-6.1-terra"},
                    subentry_id="test",
                    title="ChatGPT AI Task",
                ),
            )
            entity.hass = hass
            entity.entity_id = "ai_task.chatgpt_ai_task"
            task = ai_task.GenDataTask(
                "Opis zdjęcia", "Opisz zdjęcie.", structure, [attachment]
            )
            chat_log = conversation.ChatLog(
                hass,
                "task-session",
                content=[
                    conversation.SystemContent("You are a Home Assistant expert."),
                    conversation.UserContent(task.instructions, task.attachments),
                ],
            )
            if expected is None:
                with pytest.raises(HomeAssistantError, match="does not match"):
                    await entity._async_generate_data(task, chat_log)
            else:
                result = await entity._async_generate_data(task, chat_log)
                assert result.as_dict() == {
                    "conversation_id": "task-session",
                    "data": expected,
                }

    asyncio.run(run())


def test_ai_task_image_generation_is_opt_in():
    entry = SimpleNamespace(data={CONF_PROVIDER: PROVIDER_OPENAI_CODEX})
    subentry = SimpleNamespace(data={}, subentry_id="test", title="ChatGPT AI Task")
    feature = ai_task.AITaskEntityFeature.GENERATE_IMAGE
    assert feature not in CodexAITaskEntity(entry, subentry).supported_features
    subentry.data[CONF_CODEX_IMAGE_GENERATION] = True
    assert feature in CodexAITaskEntity(entry, subentry).supported_features


@pytest.mark.parametrize("operation", ["generations", "edits", "denied", "invalid"])
def test_codex_ai_task_native_image_request(tmp_path, operation):
    image = io.BytesIO()
    Image.new("RGB", (16, 8)).save(image, format="PNG")
    image_bytes = image.getvalue()
    encoded = base64.b64encode(image_bytes).decode()
    attachments = None
    if operation == "edits":
        path = tmp_path / "reference.png"
        path.write_bytes(image_bytes)
        attachments = [
            conversation.Attachment("media-source://test/image", "image/png", path)
        ]

    def respond(request):
        endpoint = "edits" if attachments else "generations"
        assert request.url.path == f"/backend-api/codex/images/{endpoint}"
        assert request.headers["authorization"] == "Bearer test-token"
        assert request.headers["chatgpt-account-id"] == "test-account"
        assert request.headers["x-codex-image-turn-id"]
        payload = {"prompt": "Czerwony kwadrat", "model": "gpt-image-2", "n": 1}
        if attachments:
            payload["images"] = [{"image_url": f"data:image/png;base64,{encoded}"}]
        assert json.loads(request.content) == payload
        if operation == "denied":
            return httpx.Response(
                403, json={"detail": "Image generation is unavailable"}
            )
        if operation == "invalid":
            return httpx.Response(200, json={"data": [{"b64_json": "wrong"}]})
        return httpx.Response(200, json={"data": [{"b64_json": encoded}]})

    async def run():
        hass = HomeAssistant(str(tmp_path))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            codex = OpenAICodexClient(
                "test-token", "test-account", "https://example.test/backend-api", client
            )
            entity = CodexAITaskEntity(
                SimpleNamespace(
                    data={CONF_PROVIDER: PROVIDER_OPENAI_CODEX}, runtime_data=codex
                ),
                SimpleNamespace(
                    data={CONF_CODEX_IMAGE_GENERATION: True},
                    subentry_id="test",
                    title="ChatGPT AI Task",
                ),
            )
            entity.hass = hass
            task = ai_task.GenImageTask("Ilustracja", "Czerwony kwadrat", attachments)
            chat_log = conversation.ChatLog(hass, "image-session")
            if operation in ("denied", "invalid"):
                message = (
                    "HTTP 403: Image generation is unavailable"
                    if operation == "denied"
                    else "invalid image"
                )
                with pytest.raises(HomeAssistantError, match=message):
                    await entity._async_generate_image(task, chat_log)
            else:
                result = await entity._async_generate_image(task, chat_log)
                assert result.image_data == image_bytes
                assert (result.mime_type, result.width, result.height) == (
                    "image/png",
                    16,
                    8,
                )
                assert result.conversation_id == "image-session"

    asyncio.run(run())
