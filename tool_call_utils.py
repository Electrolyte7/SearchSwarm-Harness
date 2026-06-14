"""Shared normalization helpers for model-generated tool arguments."""


def normalize_tool_args(tool_args):
    """Accept the common ``{"params": {...}}`` compatibility wrapper."""
    if not isinstance(tool_args, dict):
        return tool_args
    if set(tool_args) == {"params"} and isinstance(tool_args["params"], dict):
        return dict(tool_args["params"])
    return dict(tool_args)
