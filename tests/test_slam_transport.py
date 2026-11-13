from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from exp_workspace.lhnav_es.mcp._slam import OrbSlam3Client


def test_track_uses_contiguous_binary_payload_without_base64() -> None:
    client = object.__new__(OrbSlam3Client)
    captured = {}

    def rpc(msg, timeout_s=None, payload=()):
        captured["msg"] = msg
        captured["payload"] = payload
        return {"pose": None, "state": 1, "ok_frames": 0}

    client._rpc = rpc
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)[:, ::2]
    depth = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)[:, ::2]

    result = client.track(rgb, depth, 1.25)

    msg = captured["msg"]
    assert msg == {
        "cmd": "track", "h": 4, "w": 3, "t": 1.25,
        "rgb_nbytes": 4 * 3 * 3, "depth_nbytes": 4 * 3 * 4,
    }
    assert "rgb" not in msg and "depth" not in msg
    rgb_payload, depth_payload = captured["payload"]
    np.testing.assert_array_equal(
        np.frombuffer(rgb_payload, dtype=np.uint8).reshape(4, 3, 3), rgb
    )
    np.testing.assert_array_equal(
        np.frombuffer(depth_payload, dtype=np.float32).reshape(4, 3), depth
    )
    assert result == {"pose": None, "state": 1, "ok_frames": 0}


def test_binary_payload_header_is_small_json() -> None:
    header = json.dumps({
        "cmd": "track", "h": 480, "w": 640, "t": 0.1,
        "rgb_nbytes": 640 * 480 * 3, "depth_nbytes": 640 * 480 * 4,
    })
    raw_nbytes = 640 * 480 * 3 + 640 * 480 * 4
    assert len(header) < 200
    assert len(header) + raw_nbytes < raw_nbytes * 1.01


def test_server_reads_binary_track_frame() -> None:
    server = Path(__file__).parents[1] / "exp_workspace/lhnav_es/mcp/_slam_server.py"
    harness = r'''
import runpy, sys, types
fake = types.ModuleType("orbslam3")
class System:
    def __init__(self, *args): pass
    def set_use_viewer(self, value): pass
    def initialize(self): pass
    def process_image_rgbd(self, rgb, depth, timestamp):
        assert rgb.shape == (2, 3, 3) and rgb.dtype.name == "uint8"
        assert depth.shape == (2, 3) and depth.dtype.name == "float32"
        assert int(rgb[1, 2, 1]) == 16 and float(depth[1, 2]) == 5.0
        assert timestamp == 1.25
    def get_tracking_state(self): return 1
    def shutdown(self): pass
fake.System = System
fake.Sensor = types.SimpleNamespace(RGBD=0)
sys.modules["orbslam3"] = fake
runpy.run_path(sys.argv[1], run_name="__main__")
'''
    rgb = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.arange(6, dtype=np.float32).reshape(2, 3)
    requests = [
        json.dumps({"cmd": "init", "intr": {
            "fx": 1, "fy": 1, "cx": 1, "cy": 1, "width": 3, "height": 2,
        }}).encode() + b"\n",
        json.dumps({"cmd": "track", "h": 2, "w": 3, "t": 1.25,
                    "rgb_nbytes": rgb.nbytes, "depth_nbytes": depth.nbytes}).encode()
        + b"\n" + rgb.tobytes() + depth.tobytes(),
        b'{"cmd":"shutdown"}\n',
    ]
    proc = subprocess.run(
        [sys.executable, "-c", harness, str(server)],
        input=b"".join(requests), capture_output=True, timeout=15,
    )

    assert proc.returncode == 0, proc.stderr.decode(errors="replace")
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert replies == [
        {"ok": True},
        {"ok": True, "pose": None, "state": 1, "ok_frames": 0},
        {"ok": True},
    ]
