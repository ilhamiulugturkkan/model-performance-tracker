import copy
import json


def _dig(obj, *path):
    """Follow a path of keys through nested dicts. Returns None if anything along the way is missing."""
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _as_json(raw):
    return json.dumps(raw, separators=(",", ":"), ensure_ascii=False)


"""
=== ANTHROPIC ===
{
  "id": "msg_011Cgm1zyd2bNYGoTffkevdf",
  "container": null,
  "content": [
    {
      "citations": null,
      "text": "391",
      "type": "text"
    }
  ],
  "diagnostics": null,
  "model": "claude-haiku-5-5",
  "role": "assistant",
  "stop_details": null,
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "inference_geo": "global",
    "input_tokens": 25,
    "output_tokens": 3,
    "output_tokens_details": {
      "thinking_tokens": 0
    },
    "server_tool_use": null,
    "service_tier": "standard",
    "speed": "standard"
  }
}
"""


def parse_anthropic(raw):
    # Take the text from the blocks whose type is "text" (thinking blocks have no text).
    blocks = [b for b in (raw.get("content") or []) if b.get("type") == "text"]
    text = "".join(b.get("text", "") for b in blocks) if blocks else None

    return {
        "text": text,
        "input_tokens": _dig(raw, "usage", "input_tokens"),
        "output_tokens": _dig(raw, "usage", "output_tokens"),
        "reasoning_tokens": _dig(
            raw, "usage", "output_tokens_details", "thinking_tokens"
        ),
        "finish_reason": raw.get("stop_reason"),
        "returned_model": raw.get("model"),
        "request_id": raw.get("id"),
        "raw_response": _as_json(raw),
    }


def parse_openai(raw):
    # The text lives in "message" items, in parts of type "output_text".
    # Other items (for example "reasoning") come first or not at all, so never use a position.
    pieces = []
    for item in raw.get("output") or []:
        if item.get("type") == "message":
            pieces += [
                part.get("text", "")
                for part in (item.get("content") or [])
                if part.get("type") == "output_text"
            ]
    text = "".join(pieces) if pieces else None

    # "completed", or "incomplete:<reason>" when the reply was cut short.
    status = raw.get("status")
    reason = _dig(raw, "incomplete_details", "reason")
    finish_reason = f"{status}:{reason}" if reason else status

    # Keep the whole response, but not the opaque reasoning blob. Work on a copy.
    stored = copy.deepcopy(raw)
    for item in stored.get("output") or []:
        blob = item.get("encrypted_content")
        if isinstance(blob, str):
            item["encrypted_content"] = f"<removed {len(blob)} chars>"

    return {
        "text": text,
        "input_tokens": _dig(raw, "usage", "input_tokens"),
        "output_tokens": _dig(raw, "usage", "output_tokens"),
        "reasoning_tokens": _dig(
            raw, "usage", "output_tokens_details", "reasoning_tokens"
        ),
        "finish_reason": finish_reason,
        "returned_model": raw.get("model"),
        "request_id": raw.get("id"),
        "raw_response": _as_json(stored),
    }
