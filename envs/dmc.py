"""DeepMind Control Suite with the ground-truth hooks of plan Section 3.3.

`state()` saves MuJoCo's full integration state (mjSTATE_INTEGRATION: time,
qpos, qvel, act, warm-start accelerations, controls, ...), not just qpos and
qvel.  The solver warm-start is part of what determines the next step, so a
restore that omits it drifts at round-off level and compounds; with it, a
restored branch is bit-identical to the original.  DMC dynamics carry no
randomness after reset, so the matched-intervention and plan-cost audits see
identical conditions on every branch.
"""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
from dm_control import suite

TASKS = {"cartpole-swingup": ("cartpole", "swingup"),
         "walker-walk": ("walker", "walk"),
         "cup-catch": ("ball_in_cup", "catch"),
         # side paper Section 16B: the other three PlaNet benchmark tasks
         "reacher-easy": ("reacher", "easy"),
         "cheetah-run": ("cheetah", "run"),
         "finger-spin": ("finger", "spin")}
_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class DMCEnv:
    def __init__(self, name: str, action_repeat: int = 2, img: int = 64,
                 camera_id: int = 0, render: bool = True):
        dom, task = TASKS[name]
        self.name, self.repeat, self.img, self.cam = name, action_repeat, img, camera_id
        self._dom, self._task, self.render_on = dom, task, render
        self.env = None

    def _load(self, seed):
        self.env = suite.load(self._dom, self._task, task_kwargs={"random": int(seed)})
        self.physics = self.env.physics
        spec = self.env.action_spec()
        self.act_dim, self.a_lo, self.a_hi = spec.shape[0], spec.minimum, spec.maximum

    def _obs(self, ts):
        prop = np.concatenate([np.asarray(v, np.float32).ravel() for v in ts.observation.values()])
        out = {"proprio": prop, "qpos": self.physics.data.qpos.copy(),
               "qvel": self.physics.data.qvel.copy()}
        if self.render_on:
            out["pixels"] = self.physics.render(self.img, self.img, camera_id=self.cam)
        return out

    def reset(self, seed: int):
        self._load(seed)
        return self._obs(self.env.reset())

    def step(self, a):
        a = np.clip(np.asarray(a, np.float64), self.a_lo, self.a_hi)
        r = 0.0
        for _ in range(self.repeat):
            ts = self.env.step(a)
            r += ts.reward or 0.0
            if ts.last():
                break
        return self._obs(ts), float(r), bool(ts.last())

    # --- ground-truth hooks (plan Section 3.3) ------------------------------
    def state(self) -> np.ndarray:
        m, d = self.physics.model.ptr, self.physics.data.ptr
        buf = np.empty(mujoco.mj_stateSize(m, _SPEC))
        mujoco.mj_getState(m, d, buf, _SPEC)
        return buf

    def set_state(self, s: np.ndarray):
        m, d = self.physics.model.ptr, self.physics.data.ptr
        mujoco.mj_setState(m, d, np.asarray(s, np.float64), _SPEC)
        mujoco.mj_forward(m, d)
        ts_obs = self.env.task.get_observation(self.physics)
        class _TS:  # minimal timestep for _obs
            observation = ts_obs
        return self._obs(_TS)

    def true_state_vector(self):
        """What probes are fit to (plan Appendix B): qpos and qvel."""
        return np.concatenate([self.physics.data.qpos, self.physics.data.qvel]).astype(np.float32)
