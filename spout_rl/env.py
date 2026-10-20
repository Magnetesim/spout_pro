"""ctypes wrapper around csrc/libspout.so (batched headless Spout)."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CSRC = ROOT / "csrc"
LIB_PATH = CSRC / "libspout.so"

PLAY_H, W, SCREEN_H = 78, 128, 88
NUM_ACTIONS = 6  # action = rot + 3 * thrust, rot: 0 none, 1 left, 2 right
NUM_SCALARS = 8
NUM_INFO = 8
INFO_FIELDS = ("return", "height", "score", "frames", "mode", "start_height", "barriers", "end")

START_NORMAL, START_DRILL, START_RANDOM = 0, 1, 2
MODE_NAMES = {START_NORMAL: "normal", START_DRILL: "drill", START_RANDOM: "random"}

# Original spout palette: white, light grey, dark grey, black
PALETTE = np.array([[255, 255, 255], [170, 170, 170], [85, 85, 85], [0, 0, 0]], dtype=np.uint8)


def action_of(left: bool, right: bool, thrust: bool) -> int:
    rot = 1 if left and not right else 2 if right and not left else 0
    return rot + 3 * int(thrust)


class SpvConfig(ctypes.Structure):
    _fields_ = [
        ("frameskip", ctypes.c_int32),
        ("max_frames", ctypes.c_int32),
        ("r_height", ctypes.c_float),
        ("r_bonus", ctypes.c_float),
        ("r_barrier", ctypes.c_float),
        ("r_death", ctypes.c_float),
        ("r_hit_above", ctypes.c_float),
        ("p_drill", ctypes.c_float),
        ("p_random", ctypes.c_float),
        ("random_height_max", ctypes.c_int32),
        ("drill_max_barrier", ctypes.c_int32),
    ]


DEFAULT_CONFIG = dict(
    frameskip=4,
    max_frames=50 * 60 * 10,  # 10 minutes of game time
    r_height=0.1,  # per pixel climbed
    r_bonus=0.01,  # per point of the game's time bonus (10 points per second left)
    r_barrier=5.0,  # per barrier broken through
    r_death=-5.0,
    r_hit_above=0.003,  # shaping: per hp of rock knocked off above the ship
    p_drill=0.3,
    p_random=0.2,
    random_height_max=1500,
    drill_max_barrier=8,
)


def _needs_build() -> bool:
    if not LIB_PATH.exists():
        return True
    mtime = LIB_PATH.stat().st_mtime
    return any(p.stat().st_mtime > mtime for p in CSRC.glob("*.[ch]"))


def _load():
    if _needs_build():
        subprocess.run(["make", "-C", str(CSRC)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(LIB_PATH))
    P = ctypes.c_void_p
    lib.spv_create.restype = P
    lib.spv_create.argtypes = [ctypes.c_int32, ctypes.c_uint64, ctypes.POINTER(SpvConfig)]
    lib.spv_free.argtypes = [P]
    lib.spv_set_config.argtypes = [P, ctypes.POINTER(SpvConfig)]
    lib.spv_reset_all.argtypes = [P, P, P]
    lib.spv_start.argtypes = [P, ctypes.c_int32, ctypes.c_uint64] + [ctypes.c_int32] * 5 + [P, P]
    lib.spv_step.argtypes = [P] * 8
    lib.spv_get_state.argtypes = [P, ctypes.c_int32, P]
    lib.spv_set_threads.argtypes = [P, ctypes.c_int32]
    return lib


_lib = None


def lib():
    global _lib
    if _lib is None:
        _lib = _load()
    return _lib


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.c_void_p)


class SpoutVec:
    """n Spout games stepped together. Finished games restart automatically."""

    def __init__(self, n: int, seed: int = 0, record_frames: bool = False, threads: int = 8,
                 buffers: dict | None = None, **cfg):
        """buffers: optional preallocated arrays (e.g. views of pinned torch
        tensors) for obs, scalars, rewards, dones, so results land there directly."""
        self.n = n
        self.cfg = {**DEFAULT_CONFIG, **cfg}
        self._c = SpvConfig(**self.cfg)
        self._h = lib().spv_create(n, seed, ctypes.byref(self._c))
        self.obs = np.zeros((n, PLAY_H, W), np.uint8)
        self.scalars = np.zeros((n, NUM_SCALARS), np.float32)
        self.rewards = np.zeros(n, np.float32)
        self.dones = np.zeros(n, np.uint8)
        self.info = np.zeros((n, NUM_INFO), np.float32)
        self.record_frames = record_frames
        self.frames = (
            np.zeros((n, self.cfg["frameskip"], SCREEN_H, W), np.uint8) if record_frames else None
        )
        self._actions = np.zeros(n, np.int32)
        for k, a in (buffers or {}).items():
            cur = getattr(self, k)
            assert a.shape == cur.shape and a.dtype == cur.dtype and a.flags.c_contiguous, k
            setattr(self, k, a)
        lib().spv_set_threads(self._h, threads)

    def set_config(self, **kw):
        assert "frameskip" not in kw or kw["frameskip"] == self.cfg["frameskip"]
        self.cfg.update(kw)
        self._c = SpvConfig(**self.cfg)
        lib().spv_set_config(self._h, ctypes.byref(self._c))

    def reset(self):
        lib().spv_reset_all(self._h, _ptr(self.obs), _ptr(self.scalars))
        return self.obs, self.scalars

    def start(self, i: int, seed: int, mode: int = START_NORMAL, height: int = 0, x: int = 64,
              y: int = 58, mR: int = 768):
        lib().spv_start(self._h, i, seed, mode, height, x, y, mR,
                        _ptr(self.obs[i]), _ptr(self.scalars[i]))

    def step(self, actions):
        self._actions[:] = actions
        lib().spv_step(self._h, _ptr(self._actions), _ptr(self.obs), _ptr(self.scalars),
                       _ptr(self.rewards), _ptr(self.dones), _ptr(self.info),
                       _ptr(self.frames) if self.record_frames else None)
        return self.obs, self.scalars, self.rewards, self.dones, self.info

    def state(self, i: int) -> dict:
        out = np.zeros(10, np.int64)
        lib().spv_get_state(self._h, i, _ptr(out))
        keys = ("x", "y", "vx", "vy", "mR", "height", "score", "time", "gameover", "grains")
        return dict(zip(keys, out.tolist()))

    def close(self):
        if self._h:
            lib().spv_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def to_rgb(screen: np.ndarray, zoom: int = 4) -> np.ndarray:
    """(H, W) colors 0..3 -> (H*zoom, W*zoom, 3) RGB."""
    img = PALETTE[screen]
    if zoom > 1:
        img = img.repeat(zoom, axis=0).repeat(zoom, axis=1)
    return img
