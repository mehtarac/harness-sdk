from typing_extensions import assert_type

from strands.experimental.bidi.types import (
    BidiOutputEvent,
    BidiToolUsesCompleteEvent,
    ToolUseStreamEvent,
)
from strands.types.content import Message


def completed_tool_uses_are_public_output_events(message: Message) -> None:
    event = BidiToolUsesCompleteEvent(message)
    output_event: BidiOutputEvent = event

    assert_type(event, BidiToolUsesCompleteEvent)
    assert_type(event.message, Message)
    assert_type(output_event, BidiOutputEvent)
    assert_type(ToolUseStreamEvent(delta={}, current_tool_use={}), ToolUseStreamEvent)
