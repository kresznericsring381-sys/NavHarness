"""Host-side client for the containerized ORB-SLAM3 server.

Spawns ``docker run -i navharness/orbslam3 python3 /work/_slam_server.py``
with this directory mounted read-only at /work, and speaks the JSON-lines
protocol plus binary RGB-D payloads over the container's stdin/stdout. The
image is built from ``docker/orbslam3/``; ORB-SLAM3 is CPU-only, no GPU flags
needed.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

TRACKING_OK = 2
TRACKING_LOST = 4



def _stderr_sink():
    """Where the container's own words go. A file, appended, never discarded."""
    path = os.environ.get("ORBSLAM3_LOG") or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), f"orbslam3-{os.getpid()}.log")
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        return open(path, "a", buffering=1)          # noqa: SIM115 — owned by the process
    except OSError:
        return subprocess.DEVNULL                     # a log is not worth an episode

# How long one RPC may take before the container counts as gone. A tracked
# frame is milliseconds; loop closure and map merging can take seconds; nothing
# legitimate takes minutes. Startup loads the vocabulary, so it gets its own.
RPC_TIMEOUT_S = 180.0
INIT_TIMEOUT_S = 180.0
VOCAB_CONTAINER_PATH = "/opt/orbslam3/ORBvoc.txt"


def _assert_no_active_slam(image: str) -> None:
    out = subprocess.run(
        ["docker", "ps", "--filter", f"ancestor={image}", "--format", "{{.ID}} {{.Names}}"],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip()
    if out:
        raise SlamDied(f"refusing to start another ORB-SLAM3 container while one is active: {out[:300]}")


def _vocabulary_path(image: str = "navharness/orbslam3:latest") -> Path:
    """Copy the large text vocabulary out of Docker's overlay layer once.

    ORB-SLAM3's loader performs many small reads. A host-cached bind mount
    avoids repeating those reads through the image's overlay layer per run.
    """
    configured = os.environ.get("ORBSLAM3_VOCAB_PATH")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ORBSLAM3_VOCAB_PATH does not exist: {path}")
        return path

    image_id = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True, capture_output=True, text=True, timeout=15,
    ).stdout.strip().replace(":", "-")
    cache = Path(os.environ.get("ORBSLAM3_CACHE_DIR", "/mnt/ssd/fy/.cache/navharness/orbslam3")) / image_id
    target = cache / "ORBvoc.txt"
    if target.is_file() and target.stat().st_size > 100_000_000:
        return target

    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / f"ORBvoc.txt.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    container = f"navh-orbvoc-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    try:
        subprocess.run(["docker", "create", "--name", container, image],
                       check=True, capture_output=True, text=True, timeout=30)
        subprocess.run(["docker", "cp", f"{container}:{VOCAB_CONTAINER_PATH}", str(tmp)],
                       check=True, capture_output=True, text=True, timeout=120)
        if tmp.stat().st_size < 100_000_000:
            raise RuntimeError(f"ORB-SLAM3 vocabulary copy is unexpectedly small: {tmp.stat().st_size} bytes")
        tmp.replace(target)
        return target
    finally:
        tmp.unlink(missing_ok=True)
        with contextlib.suppress(Exception):
            subprocess.run(["docker", "rm", "-f", container], capture_output=True, timeout=30)


class SlamError(RuntimeError):
    """ORB-SLAM3 refused a call (an exception inside the library, reported
    back by the server, e.g. ``Invalid argument``). The process is alive but
    its state is not to be trusted; the caller treats it like a dead one."""


class SlamDied(RuntimeError):
    """The ORB-SLAM3 container is gone. Seen when its own loop closure throws
    (``terminate called ... cv::Exception``) — a C++ death this process
    cannot prevent, only notice. The caller may open a new session."""


class OrbSlam3Client:
    def __init__(self, image: str = "navharness/orbslam3:latest") -> None:
        _assert_no_active_slam(image)
        vocab = _vocabulary_path(image)
        self.container_name = f"navh-orbslam-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self._proc = subprocess.Popen(
            [
                "docker", "run", "-i", "--rm", "--name", self.container_name,
                "-v", f"{_THIS_DIR}:/work:ro",
                "-v", f"{vocab}:{VOCAB_CONTAINER_PATH}:ro",
                image, "python3", "/work/_slam_server.py",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # NOT DEVNULL. ORB-SLAM3 is a C++ library in a container: when it
            # dies, what it said on the way out is the only evidence there is.
            # Set ORBSLAM3_LOG to keep the file somewhere else.
            stderr=_stderr_sink(),
        )

    def _rpc(self, msg: dict, timeout_s: float | None = None,
             payload: tuple[memoryview, ...] = ()) -> dict:
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(json.dumps(msg).encode("utf-8") + b"\n")
            for part in payload:
                self._proc.stdin.write(part)
            self._proc.stdin.flush()
            # A container that stops ANSWERING is as dead as one that exits, and
            # far more expensive: a hung frame holds the scene until the env's
            # own timeout. Wait a bounded time, then take the session as gone.
            wait = RPC_TIMEOUT_S if timeout_s is None else timeout_s
            if wait and not select.select([self._proc.stdout], [], [], wait)[0]:
                with contextlib.suppress(Exception):
                    self._proc.kill()
                raise SlamDied(f"slam_server did not answer {msg.get('cmd')} within {wait:.0f}s")
            line = self._proc.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            raise SlamDied(f"slam_server is gone: {type(e).__name__}: {e}") from e
        if not line:
            raise SlamDied("slam_server closed its stdout (the container died)")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise SlamError(f"slam_server error: {resp.get('error')}")
        return resp

    def init(self, intrinsics: dict, fps: float = 10.0) -> None:
        intr = dict(intrinsics)
        intr["fps"] = fps
        started = time.monotonic()
        print("[slam] ORB-SLAM3 init_start", flush=True)
        try:
            self._rpc({"cmd": "init", "intr": intr}, timeout_s=INIT_TIMEOUT_S)
        except BaseException as exc:
            print(f"[slam] ORB-SLAM3 init_failed after {time.monotonic() - started:.1f}s: {exc}", flush=True)
            with contextlib.suppress(Exception):
                self._proc.kill()
            with contextlib.suppress(Exception):
                subprocess.run(["docker", "rm", "-f", self.container_name],
                               capture_output=True, timeout=30)
            raise
        print(f"[slam] ORB-SLAM3 init ready in {time.monotonic() - started:.1f}s", flush=True)

    def track(self, rgb: np.ndarray, depth_m: np.ndarray, t: float) -> dict:
        """Returns {"pose": 4x4 ndarray | None, "state": int, "ok_frames": int}."""
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(depth_m, dtype=np.float32)
        h, w = depth.shape
        if rgb.shape != (h, w, 3):
            raise ValueError(f"RGB shape {rgb.shape} does not match depth shape {(h, w)}")
        resp = self._rpc({
            "cmd": "track",
            "h": h, "w": w, "t": float(t),
            "rgb_nbytes": rgb.nbytes, "depth_nbytes": depth.nbytes,
        }, payload=(memoryview(rgb).cast("B"), memoryview(depth).cast("B")))
        pose = resp["pose"]
        return {
            "pose": None if pose is None else np.asarray(pose, dtype=np.float64),
            "state": int(resp["state"]),
            "ok_frames": int(resp["ok_frames"]),
        }

    def trajectory(self) -> list:
        return self._rpc({"cmd": "trajectory"})["trajectory"]

    def keyframes(self) -> list:
        """Every atlas map's keyframes, as ORB-SLAM3 currently believes them.

        The one thing a frame-by-frame client cannot see on its own: when a
        loop closes, or when two maps are merged, the poses of keyframes
        already reported change. Reading this is how the map on our side gets
        to hear about it.
        """
        return self._rpc({"cmd": "keyframes"})["keyframes"]

    def close(self) -> None:
        with contextlib.suppress(Exception):  # teardown must not raise
            self._rpc({"cmd": "shutdown"})
        with contextlib.suppress(Exception):
            self._proc.stdin.close()  # type: ignore[union-attr]
        try:
            self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            with contextlib.suppress(Exception):
                subprocess.run(["docker", "rm", "-f", self.container_name],
                               capture_output=True, timeout=30)
