from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.llm.providers.openai import OpenAIAdapter
from agent.llm.types import Message, StreamDone, TextBlock, ThinkingBlock

LEAKED_MARKUP = (
    "<｜｜DSML｜｜tool_calls>\n"
    '<｜｜DSML｜｜invoke name="run_command">\n'
    '<｜｜DSML｜｜parameter name="command" string="true">'
    'python -c "import sys; print(...)"</｜｜DSML｜｜parameter>\n'
    "</｜｜DSML｜｜invoke>\n"
    "</｜｜DSML｜｜tool_calls>"
)


def _fake_response(content: str) -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _parse_text_blocks(content: str) -> list[TextBlock]:
    response = _fake_response(content)
    result = OpenAIAdapter.parse_response(response)
    text_blocks = [b for b in result.content if isinstance(b, TextBlock)]
    return text_blocks


def test_parse_response_strips_leaked_tool_markup_keeps_surrounding_text() -> None:
    content = f"Let me check that.\n{LEAKED_MARKUP}\nDone."
    text_blocks = _parse_text_blocks(content)

    assert len(text_blocks) == 1
    assert "｜｜DSML｜｜" not in text_blocks[0].text
    assert "Let me check that." in text_blocks[0].text
    assert "Done." in text_blocks[0].text


def test_parse_response_drops_text_block_when_only_leaked_markup() -> None:
    text_blocks = _parse_text_blocks(LEAKED_MARKUP)

    assert text_blocks == []


def test_parse_response_plain_text_unaffected() -> None:
    text_blocks = _parse_text_blocks("Just a normal answer, no markup here.")

    assert len(text_blocks) == 1
    assert text_blocks[0].text == "Just a normal answer, no markup here."


def test_translate_messages_thinking_only_assistant_message_sets_empty_content() -> None:
    message = Message(role="assistant", content=[ThinkingBlock(text="some reasoning")])

    result = OpenAIAdapter.translate_messages([message])

    assert len(result) == 1
    entry = result[0]
    assert entry["content"] == ""
    assert entry["reasoning_content"] == "some reasoning"
    assert "tool_calls" not in entry


def _make_adapter() -> OpenAIAdapter:
    with patch("openai.AsyncOpenAI"):
        return OpenAIAdapter(api_key="key")


async def _fake_stream(pieces: list[str]) -> AsyncIterator[SimpleNamespace]:
    for piece in pieces:
        delta = SimpleNamespace(content=piece, reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(delta=delta, finish_reason=None)
        yield SimpleNamespace(choices=[choice], usage=None)
    delta = SimpleNamespace(content=None, reasoning_content=None, tool_calls=None)
    choice = SimpleNamespace(delta=delta, finish_reason="stop")
    yield SimpleNamespace(choices=[choice], usage=None)


def test_stream_strips_leaked_tool_markup_from_final_text() -> None:
    adapter = _make_adapter()
    adapter._client.chat.completions.create = AsyncMock(
        return_value=_fake_stream(["Let me check that.\n", LEAKED_MARKUP, "\nDone."])
    )

    async def _collect() -> list:
        events = []
        async for event in adapter.stream(model="test-model", messages=[]):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    done_events = [e for e in events if isinstance(e, StreamDone)]
    assert len(done_events) == 1
    text_blocks = [b for b in done_events[0].response.content if isinstance(b, TextBlock)]
    assert len(text_blocks) == 1
    assert "｜｜DSML｜｜" not in text_blocks[0].text
    assert "Let me check that." in text_blocks[0].text
    assert "Done." in text_blocks[0].text
