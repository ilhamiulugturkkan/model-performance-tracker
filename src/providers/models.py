# All models' configs to evaluate

MODELS = {
    "haiku": {
        "provider": "anthropic",
        "model": "claude-haiku-5-5",
        "settings": {"max_tokens": 4096},
    },
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "settings": {"max_tokens": 4096},
    },
    "opus": {
        "provider": "anthropic",
        "model": "claude-opus-5-5",
        "settings": {"max_tokens": 4096},
    },
    "luna": {
        "provider": "openai",
        "model": "gpt-6-luna",
        "settings": {"max_tokens": 4096, "temperature": 0, "reasoning_effort": "none"},
    },
    "terra": {
        "provider": "openai",
        "model": "gpt-6.1-terra",
        "settings": {"max_tokens": 4096, "temperature": 0, "reasoning_effort": "none"},
    },
    "sol": {
        "provider": "openai",
        "model": "gpt-6-sol",
        "settings": {"max_tokens": 4096, "temperature": 0, "reasoning_effort": "none"},
    },
}


def parameters_record(config):
    """What goes in calls.parameters: the settings, plus anything that was NOT settable."""
    record = dict(config["settings"])
    if "temperature" not in record:
        record["temperature"] = "provider default (not settable)"
    return record
