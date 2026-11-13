from __future__ import annotations

import asyncio
from types import SimpleNamespace

from runner import Runner


class _PlacedModule:
    seen: list[int] = []

    def __init__(self, env, cfg, arm) -> None:
        self.env, self.cfg, self.arm = env, cfg, arm

    async def place(self, index: int):
        self.seen.append(index)
        return SimpleNamespace(
            goal=SimpleNamespace(
                episode_id="goat-1", scene_id="scene-1", text="find the object", n_goals=10
            ),
            caps={"max_turns": 120},
        )


def test_preflight_places_one_episode_without_building_a_model_session(monkeypatch) -> None:
    runner = object.__new__(Runner)
    runner.client_timeout_s = 240
    runner.module_cls = _PlacedModule
    runner.cfg = {"split": "val_unseen"}
    runner.arm = object()
    _PlacedModule.seen = []
    monkeypatch.setattr("runner.env_client_factory", lambda url, timeout_s: (url, timeout_s))

    result = asyncio.run(runner.preflight_episode("http://env", 1))

    assert _PlacedModule.seen == [1]
    assert result == {
        "status": "ready",
        "stage": "placement_and_slam",
        "episode_index": 1,
        "episode_id": "goat-1",
        "scene_id": "scene-1",
        "goal": "find the object",
        "n_goals": 10,
        "caps": {"max_turns": 120},
        "elapsed_s": result["elapsed_s"],
    }
