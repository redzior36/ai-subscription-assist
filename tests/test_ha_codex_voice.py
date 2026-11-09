"""Regression tests using real Home Assistant classes and mocked providers."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import wave

import pytest

pytest.importorskip("homeassistant")

from homeassistant.components import conversation, stt as ha_stt
from homeassistant.const import EVENT_HOMEASSISTANT_CLOSE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import llm
from homeassistant.helpers.httpx_client import get_async_client
import httpx
import voluptuous as vol

from custom_components.ai_subscription_assist import (
    OpenAIClient,
    OpenAICodexClient,
    _create_client,
    _extract_openai_codex_account_id,
    _openai_token_expiry,
)
from custom_components.ai_subscription_assist import config_flow, stt
from custom_components.ai_subscription_assist.codex_stream import async_codex_response
from custom_components.ai_subscription_assist.const import (
    CONF_ACCESS_TOKEN,
    CONF_EXPIRES_AT,
    CONF_OPENAI_CODEX_ACCOUNT_ID,
    CONF_PROVIDER,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_CODEX_MODEL,
    PROVIDER_CLAUDE_OAUTH,
    PROVIDER_OPENAI,
    PROVIDER_OPENAI_CODEX,
)
from custom_components.ai_subscription_assist.entity import (
    AiSubscriptionAssistBaseLLMEntity,
    _format_openai_responses_tool,
)
from custom_components.ai_subscription_assist.tools import AiSubscriptionAssistTool


def _event(kind, **fields):
    return f"data: {json.dumps({'type': kind, **fields})}\r\n\r\n"


class DeviceStatusTool(AiSubscriptionAssistTool):
    name = "device_status"
    parameters = vol.Schema({vol.Required("name"): str})

    async def _async_call(self, hass, tool_input, llm_context):
        return {"state": "on", "name": tool_input.tool_args["name"]}


def test_codex_conversation_replays_tool_result(tmp_path):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.headers["accept"] == "text/event-stream"
        assert request.headers["chatgpt-account-id"] == "test-account"
        assert payload["stream"] is True and payload["store"] is False
        assert payload["model"] == DEFAULT_OPENAI_CODEX_MODEL
        assert not {"temperature", "max_output_tokens", "text"} & payload.keys()
        assert payload["tools"][0]["parameters"]["properties"]["name"] == {
            "type": "string"
        }
        if len(requests) == 1:
            tool = {
                "type": "function_call",
                "name": "device_status",
                "arguments": '{"name":"lampa"}',
                "call_id": "call_test",
                "id": "fc_test",
            }
            # Include both forms to verify the tool runs only once.
            stream = _event("response.output_item.done", item=tool) + _event(
                "response.completed", response={"output": [tool], "usage": {}}
            )
        else:
            assert payload["input"][-2]["call_id"] == "call_test"
            assert "id" not in payload["input"][-2]
            assert payload["input"][-1]["type"] == "function_call_output"
            assert json.loads(payload["input"][-1]["output"]) == {
                "state": "on",
                "name": "lampa",
            }
            stream = _event(
                "response.completed",
                response={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Świeci."}],
                        }
                    ],
                },
            )
        return httpx.Response(
            200, text=stream, headers={"Content-Type": "text/event-stream"}
        )

    async def run():
        hass = HomeAssistant(str(tmp_path))
        tool = DeviceStatusTool(hass, None)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http_client:
            client = OpenAICodexClient(
                "test-token", "test-account", "https://example.test", http_client
            )
            agent = AiSubscriptionAssistBaseLLMEntity(
                SimpleNamespace(runtime_data=client, data={}),
                SimpleNamespace(data={}, subentry_id="test", title="Codex"),
            )
            agent.entity_id = "conversation.test_codex"
            chat_log = conversation.ChatLog(
                hass,
                "test",
                content=[
                    conversation.SystemContent(""),
                    conversation.UserContent("Czy lampa świeci?"),
                ],
                llm_api=llm.APIInstance(
                    api=SimpleNamespace(hass=hass),
                    api_prompt="",
                    llm_context=llm.LLMContext("test", None, "pl", "assist", None),
                    tools=[tool],
                    custom_serializer=llm.selector_serializer,
                ),
            )
            await agent._async_handle_chat_log_openai_codex(chat_log)
            assert chat_log.content[-1].content == "Świeci."
            assert len(requests) == 2

    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "body", "message"),
    [
        (400, '{"detail":"Invalid model"}', "Invalid model"),
        (401, '{"error":{"message":"secret"}}', "authentication failed"),
        (
            200,
            _event(
                "response.failed", response={"error": {"message": "Model unavailable"}}
            ),
            "Model unavailable",
        ),
        (
            200,
            _event("response.output_item.done", item={"type": "function_call"}),
            "before the response completed",
        ),
        (200, "data: invalid json\n\n", "invalid SSE data"),
    ],
)
def test_codex_stream_errors(status, body, message):
    async def run():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status, text=body)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(HomeAssistantError, match=message):
                await async_codex_response(client, "https://example.test", {}, {})

    asyncio.run(run())


def test_codex_sse_multiline_and_item_fallback():
    async def run():
        stream = (
            ": heartbeat\n\n"
            + _event("response.output_item.done", item={"type": "message"})
            + 'event: response.completed\ndata: {"type":"response.completed",\ndata: "response":{"usage":{}}}\n\n'
        )
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, text=stream)
        )
        async with httpx.AsyncClient(transport=transport) as client:
            response = await async_codex_response(
                client, "https://example.test", {}, {}
            )
        assert response == {"output": [{"type": "message"}], "usage": {}}

    asyncio.run(run())


def test_oauth_token_claim_fallback():
    claims = {
        "exp": 4102444800,
        "https://api.openai.com/auth": {"chatgpt_account_id": "account"},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token = f"header.{encoded}.signature"
    assert _openai_token_expiry({"access_token": token}) == claims["exp"]
    assert _extract_openai_codex_account_id("invalid", token) == "account"


def test_http_clients_use_matching_transports(tmp_path):
    async def run():
        hass = HomeAssistant(str(tmp_path))
        client = get_async_client(hass)
        assert isinstance(client, httpx.AsyncClient)
        claude = _create_client(hass, "test-token")
        assert claude._client is _create_client(hass, "test-token")._client
        hass.bus.async_fire(EVENT_HOMEASSISTANT_CLOSE)
        await hass.async_block_till_done()
        assert client.is_closed

    asyncio.run(run())


def test_tool_schema_and_validation():
    tool = DeviceStatusTool(None, None)
    definition = _format_openai_responses_tool(tool, llm.selector_serializer)
    assert json.loads(json.dumps(definition))["parameters"]["required"] == ["name"]
    with pytest.raises(vol.Invalid):
        asyncio.run(tool.async_call(None, llm.ToolInput("device_status", {}), None))


def test_claude_stt_polish_and_provider_selection(monkeypatch):
    async def fake_transcribe(session, token, stream, *, language, keyterms):
        assert token == "test-token"
        assert language == "pl"
        assert [chunk async for chunk in stream] == [b"\0\0"]
        return "włącz lampę"

    async def audio():
        yield b"\0\0"

    async def run():
        add = Mock()
        for provider in (PROVIDER_OPENAI_CODEX, PROVIDER_CLAUDE_OAUTH):
            entry = SimpleNamespace(
                entry_id=provider,
                data={
                    CONF_PROVIDER: provider,
                    CONF_ACCESS_TOKEN: "test-token",
                    CONF_EXPIRES_AT: 4102444800,
                },
            )
            await stt.async_setup_entry(None, entry, add)
        add.assert_called_once()
        engine = add.call_args.args[0][0]
        assert "pl" in engine.supported_languages
        engine.hass = Mock()
        metadata = ha_stt.SpeechMetadata(
            "pl",
            engine.supported_formats[0],
            engine.supported_codecs[0],
            engine.supported_bit_rates[0],
            engine.supported_sample_rates[0],
            engine.supported_channels[0],
        )
        result = await engine.async_process_audio_stream(metadata, audio())
        assert result.result == ha_stt.SpeechResultState.SUCCESS
        assert result.text == "włącz lampę"

    monkeypatch.setattr(stt, "async_get_clientsession", Mock())
    monkeypatch.setattr(stt, "_exposed_entity_names", Mock(return_value=[]))
    monkeypatch.setattr(stt, "async_transcribe", AsyncMock(side_effect=fake_transcribe))
    asyncio.run(run())


@pytest.mark.parametrize(
    ("status", "state"),
    [
        (200, ha_stt.SpeechResultState.SUCCESS),
        (401, ha_stt.SpeechResultState.ERROR),
    ],
)
def test_openai_stt_uploads_wav_with_polish_hint(status, state):
    pcm = b"\x01\x02" * 100

    async def audio():
        yield pcm[:3]
        yield pcm[3:]

    def respond(request):
        assert str(request.url) == f"{DEFAULT_OPENAI_BASE_URL}/audio/transcriptions"
        assert request.headers["authorization"] == "Bearer api-key"
        body = request.content
        assert b'name="model"\r\n\r\ngpt-transcribe' in body
        assert b'name="languages[]"\r\n\r\npl' in body
        wav_data = body[body.index(b"RIFF") :]
        with wave.open(io.BytesIO(wav_data), "rb") as wav:
            assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
                16000,
                1,
                2,
            )
            assert wav.readframes(wav.getnframes()) == pcm
        return httpx.Response(status, json={"text": " włącz lampę "})

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond)
        ) as http_client:
            entry = SimpleNamespace(
                entry_id="test-openai",
                data={CONF_PROVIDER: PROVIDER_OPENAI},
                runtime_data=OpenAIClient(
                    "api-key", DEFAULT_OPENAI_BASE_URL, http_client
                ),
            )
            add = Mock()
            await stt.async_setup_entry(None, entry, add)
            engine = add.call_args.args[0][0]
            assert "pl" in engine.supported_languages
            metadata = ha_stt.SpeechMetadata(
                "pl-PL",
                ha_stt.AudioFormats.WAV,
                ha_stt.AudioCodecs.PCM,
                ha_stt.AudioBitRates.BITRATE_16,
                ha_stt.AudioSampleRates.SAMPLERATE_16000,
                ha_stt.AudioChannels.CHANNEL_MONO,
            )
            result = await engine.async_process_audio_stream(metadata, audio())
            assert result.result == state
            assert result.text == ("włącz lampę" if status == 200 else None)

    asyncio.run(run())


@pytest.mark.parametrize("account", ["same-account", "another-account"])
def test_codex_reauth_keeps_existing_entry(account, monkeypatch):
    claims = {
        "exp": 4102444800,
        "https://api.openai.com/auth": {"chatgpt_account_id": account},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    token = f"header.{encoded}.signature"

    def respond(request):
        assert b"grant_type=authorization_code" in request.content
        return httpx.Response(
            200, json={"access_token": token, "refresh_token": "rotated"}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            monkeypatch.setattr(
                config_flow, "get_async_client", Mock(return_value=client)
            )
            flow = config_flow.AiSubscriptionAssistConfigFlow()
            flow.hass = Mock()
            flow.context = {"source": "reauth"}
            entry_data = {
                CONF_PROVIDER: PROVIDER_OPENAI_CODEX,
                CONF_OPENAI_CODEX_ACCOUNT_ID: "same-account",
            }
            entry = SimpleNamespace(data=entry_data)
            flow._get_reauth_entry = Mock(return_value=entry)
            flow.async_update_reload_and_abort = Mock(return_value={"type": "abort"})
            flow.async_abort = Mock(return_value={"type": "abort"})
            form = await flow.async_step_reauth(entry_data)
            assert form["step_id"] == "openai_codex"
            await flow.async_step_openai_codex({"auth_code": "sample-code"})
            if account == "same-account":
                flow.async_update_reload_and_abort.assert_called_once()
                assert flow.async_update_reload_and_abort.call_args.args == (entry,)
                assert (
                    flow.async_update_reload_and_abort.call_args.kwargs["data_updates"][
                        CONF_ACCESS_TOKEN
                    ]
                    == token
                )
            else:
                flow.async_abort.assert_called_once_with(reason="wrong_chatgpt_account")
                flow.async_update_reload_and_abort.assert_not_called()

    asyncio.run(run())
