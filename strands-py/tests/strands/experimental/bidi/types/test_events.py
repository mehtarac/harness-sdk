"""Tests for bidirectional streaming event types.

This module tests JSON serialization for all bidirectional streaming event types.
"""

import base64
import json
from typing import get_args

import pytest

from strands.experimental.bidi.types import (
    BidiAudioDeltaEvent,
    BidiAudioStartEvent,
    BidiAudioStopEvent,
    BidiBargeInEvent,
    BidiConnectionStartEvent,
    BidiConnectionStopEvent,
    BidiResponseStartEvent,
    BidiResponseStopEvent,
    BidiToolUsesCompleteEvent,
    BidiTranscriptDeltaEvent,
    BidiTranscriptStartEvent,
    BidiTranscriptStopEvent,
    BidiUsageEvent,
)
from strands.experimental.bidi.types.events import _normalize_role
from strands.types.content import Message


@pytest.mark.parametrize(
    "event_class,kwargs,expected_type",
    [
        # Output events
        (
            BidiConnectionStartEvent,
            {"connection_id": "c1", "model": "m1"},
            "bidi_connection_start",
        ),
        (BidiResponseStartEvent, {"response_id": "r1"}, "bidi_response_start"),
        (BidiTranscriptStartEvent, {"role": "user", "content_id": "u1"}, "bidi_transcript_start"),
        (
            BidiAudioStartEvent,
            {},
            "bidi_audio_start",
        ),
        (BidiAudioStopEvent, {}, "bidi_audio_stop"),
        (
            BidiAudioDeltaEvent,
            {
                "audio": base64.b64encode(b"audio").decode("utf-8"),
                "format": "pcm",
                "sample_rate": 24000,
                "channels": 1,
            },
            "bidi_audio_delta",
        ),
        (
            BidiTranscriptDeltaEvent,
            {
                "delta": "Hello",
                "role": "assistant",
                "content_id": "t1",
            },
            "bidi_transcript_delta",
        ),
        (
            BidiTranscriptStopEvent,
            {"transcript": "Hello", "role": "assistant", "content_id": "t1"},
            "bidi_transcript_stop",
        ),
        (BidiBargeInEvent, {"reason": "user_speech"}, "bidi_barge_in"),
        (
            BidiResponseStopEvent,
            {"response_id": "r1", "stop_reason": "end_turn"},
            "bidi_response_stop",
        ),
        (
            BidiToolUsesCompleteEvent,
            {
                "message": {
                    "role": "assistant",
                    "content": [{"toolUse": {"toolUseId": "call-1", "name": "weather", "input": {}}}],
                }
            },
            "bidi_tool_uses_complete",
        ),
        (
            BidiUsageEvent,
            {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "bidi_usage",
        ),
        (
            BidiConnectionStopEvent,
            {"connection_id": "c1", "reason": "complete"},
            "bidi_connection_stop",
        ),
    ],
)
def test_event_json_serialization(event_class, kwargs, expected_type):
    """Test that all event types are JSON serializable and deserializable."""
    # Create event
    event = event_class(**kwargs)

    # Verify type field
    assert event["type"] == expected_type

    # Serialize to JSON
    json_str = json.dumps(event)
    print("event_class:", event_class)
    print(json_str)
    # Deserialize back
    data = json.loads(json_str)

    # Verify type preserved
    assert data["type"] == expected_type

    # Verify all non-private keys preserved
    for key in event.keys():
        if not key.startswith("_"):
            assert key in data


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_transcript_start_contains_metadata(role):
    event = BidiTranscriptStartEvent(role, "t1")
    assert event == {"type": "bidi_transcript_start", "role": role, "content_id": "t1"}
    assert (event.role, event.content_id) == (role, "t1")


def test_audio_start_is_marker():
    start = BidiAudioStartEvent()
    assert start == {"type": "bidi_audio_start"}


def test_audio_stop_is_marker():
    stop = BidiAudioStopEvent()
    assert stop == {"type": "bidi_audio_stop"}


def test_transcript_delta_event_contains_text_delta():
    """Test that a transcript delta event contains only the incremental text."""
    event = BidiTranscriptDeltaEvent("Hello", "user", "user-transcript")

    assert event.role == "user"
    assert event.delta == "Hello"


def test_transcript_stop_event_contains_full_transcript():
    """Test that a stop event carries one authoritative transcript."""
    event = BidiTranscriptStopEvent("Hello world", "assistant", "assistant-transcript")

    exp_event = {
        "type": "bidi_transcript_stop",
        "transcript": "Hello world",
        "role": "assistant",
        "content_id": "assistant-transcript",
    }
    assert event == exp_event
    assert json.loads(json.dumps(event)) == exp_event


@pytest.mark.parametrize(
    "raw_role,expected",
    [
        ("user", "user"),
        ("assistant", "assistant"),
        ("USER", "user"),
        ("Assistant", "assistant"),
    ],
)
def test_normalize_role_accepts_supported_roles(raw_role, expected):
    """normalize_role lowercases and preserves supported roles."""
    assert _normalize_role(raw_role) == expected


@pytest.mark.parametrize(
    "raw_role",
    ["system", "admin", "SYSTEM", "tool", "", "unknown", None, 123],
)
def test_normalize_role_falls_back_to_lowest_trust_role(raw_role):
    """normalize_role coerces out-of-range values to the lowest-trust default ("user")."""
    assert _normalize_role(raw_role) == "user"
    assert _normalize_role(raw_role, default="assistant") == "assistant"


@pytest.mark.parametrize(
    "raw_role,expected",
    [
        (" user ", "user"),
        (" User ", "user"),
        ("\tassistant\n", "assistant"),
        ("  USER", "user"),
    ],
)
def test_normalize_role_strips_whitespace(raw_role, expected):
    """normalize_role trims surrounding whitespace before the allowlist check."""
    assert _normalize_role(raw_role) == expected


@pytest.mark.parametrize("raw_role", ["system", "admin", "SYSTEM", "tool", "developer", "unknown", ""])
def test_transcript_delta_event_coerces_out_of_range_role_to_user(raw_role):
    """An out-of-range transcript role is coerced to the lowest-trust role ("user")."""
    event = BidiTranscriptDeltaEvent(delta="hi", role=raw_role, content_id="transcript")

    # Attacker-controlled content is never attributed to the assistant.
    assert event.role == "user"
    assert event["role"] == "user"


def test_transcript_delta_event_strips_whitespace_role():
    """A legitimately-spaced role is trimmed rather than mislabeled as the default."""
    event = BidiTranscriptDeltaEvent(delta="hi", role=" user ", content_id="transcript")

    assert event.role == "user"


def test_transcript_delta_event_normalizes_role_casing():
    """A supported role in mixed casing is normalized to lowercase."""
    event = BidiTranscriptDeltaEvent(delta="hi", role="USER", content_id="transcript")

    assert event.role == "user"


def test_tool_uses_complete_event_preserves_provider_order():
    message: Message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "call-2", "name": "second", "input": {}}},
            {"toolUse": {"toolUseId": "call-1", "name": "first", "input": {}}},
        ],
    }

    event = BidiToolUsesCompleteEvent(message)

    assert event.message is message
    assert [tool_use["toolUseId"] for tool_use in event.tool_uses] == ["call-2", "call-1"]


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"role": "user", "content": [{"toolUse": {"toolUseId": "1", "name": "tool", "input": {}}}]}, "role"),
        ({"role": "assistant", "content": []}, "empty"),
        ({"role": "assistant", "content": [{"text": "not a tool"}]}, "only 'toolUse'"),
        ({"role": "assistant", "content": [{"toolUse": {"name": "tool", "input": {}}}]}, "toolUseId"),
        ({"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "input": {}}}]}, "name"),
        ({"role": "assistant", "content": [{"toolUse": {"toolUseId": "1", "name": "tool"}}]}, "input"),
    ],
)
def test_tool_uses_complete_event_rejects_invalid_message(message, error):
    with pytest.raises(ValueError, match=error):
        BidiToolUsesCompleteEvent(message)


def test_tool_uses_complete_event_rejects_duplicate_ids():
    message: Message = {
        "role": "assistant",
        "content": [
            {"toolUse": {"toolUseId": "duplicate", "name": "first", "input": {}}},
            {"toolUse": {"toolUseId": "duplicate", "name": "second", "input": {}}},
        ],
    }

    with pytest.raises(ValueError, match="duplicate toolUseId 'duplicate'"):
        BidiToolUsesCompleteEvent(message)


def test_tool_uses_complete_event_is_in_output_union():
    from strands.experimental.bidi.types import BidiOutputEvent

    assert BidiToolUsesCompleteEvent in get_args(BidiOutputEvent)
