from __future__ import annotations

import litellm
from openai import APIError as OpenAIAPIError
import pytest

from core.harnesses.mini.model import NavToolsModel


def test_navtools_model_uses_bounded_one_shot_api_requests() -> None:
    model = NavToolsModel(model_name="openai/example", tools=[])

    assert model._completion_kwargs() == {"timeout": 300, "num_retries": 0}
    assert model._completion_kwargs({"timeout": 45}) == {"timeout": 45, "num_retries": 0}
    assert OpenAIAPIError in model.abort_exceptions
    assert issubclass(litellm.exceptions.Timeout, OpenAIAPIError)


def test_api_timeout_is_not_replayed_by_minisweagent(monkeypatch) -> None:
    model = NavToolsModel(model_name="openai/example", tools=[])
    calls = []

    def timeout(**kwargs):
        calls.append(kwargs)
        raise litellm.exceptions.Timeout("request timed out", model="example", llm_provider="openai")

    monkeypatch.setattr(litellm, "completion", timeout)
    with pytest.raises(litellm.exceptions.Timeout):
        model.query([{"role": "user", "content": "test"}])

    assert len(calls) == 1
    assert calls[0]["timeout"] == 300
    assert calls[0]["num_retries"] == 0
