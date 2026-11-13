from __future__ import annotations

import subprocess

import pytest

from exp_workspace.lhnav_es.mcp import _slam


def test_slam_refuses_to_start_a_second_container(monkeypatch) -> None:
    def list_active(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, stdout="abc navh-orbslam-stale\n", stderr="")

    monkeypatch.setattr(_slam.subprocess, "run", list_active)

    with pytest.raises(_slam.SlamDied, match="one is active"):
        _slam._assert_no_active_slam("navharness/orbslam3:latest")


def test_vocab_override_avoids_docker_copy(tmp_path, monkeypatch) -> None:
    vocab = tmp_path / "ORBvoc.txt"
    vocab.write_text("vocabulary")
    monkeypatch.setenv("ORBSLAM3_VOCAB_PATH", str(vocab))

    assert _slam._vocabulary_path("unused-image") == vocab


def test_environment_initialize_configures_dataset_without_placing_episode(monkeypatch) -> None:
    from exp_workspace.lhnav_es.mcp.env import SlamEsEnvManager

    manager = SlamEsEnvManager()
    monkeypatch.setattr(manager, "list_splits", lambda line: ["val_unseen"])

    def build_stack() -> None:
        manager._order = [4, 8]

    monkeypatch.setattr(manager, "_ensure_built_unlocked", build_stack)
    monkeypatch.setattr(manager, "get_episode_info", lambda: pytest.fail("configure must not place an episode"))
    try:
        result = manager.initialize(line="goat", split="val_unseen", instruments=2)
    finally:
        manager.executor.shutdown(wait=True)

    assert result == {
        "line": "goat", "split": "val_unseen", "instruments": 2, "episodes": 2, "placed": False
    }
