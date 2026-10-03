"""Fake api handler of the wrapper tests, driven by the host `extras`."""

from typing import Any


def fetch(request: Any) -> dict:
    """Fail with `fail`, or write `content` (or the `token_env` secret)."""
    extras = request.host["extras"]
    if "fail" in extras:
        raise RuntimeError(extras["fail"])
    content = (
        request.secret("token_env") if "token_env" in extras else extras["content"]
    )
    request.destination.write_text(content)
    return {"handler": "fake", "placeholders": request.placeholders}
