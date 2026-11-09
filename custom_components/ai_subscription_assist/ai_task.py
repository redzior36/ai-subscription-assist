"""Home Assistant AI Task support for ChatGPT subscriptions."""

from __future__ import annotations

import json

import voluptuous as vol

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import AiSubscriptionAssistConfigEntry
from .const import SUBENTRY_AI_TASK
from .entity import AiSubscriptionAssistBaseLLMEntity

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

    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

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
