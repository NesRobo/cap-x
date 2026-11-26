"""Wrapper around capx/integrations/robosuite_cubes.py that adds a human proxy and SSM.

- StackWithHuman: robosuite Stack task with a visual-only capsule ("human proxy") added to the world.
- FrankaRobosuiteCubesHumanLowLevel: capx low-level env whose move_to_joints_blocking() limits
  joint speed according to the robot<->human distance and stops when too close.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import viser.transforms as vtf
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.environments.manipulation.stack import Stack
from robosuite.utils.placement_samplers import UniformRandomSampler

from capx.envs.simulators.robosuite_base import RobosuiteBaseEnv
from capx.envs.simulators.robosuite_cubes import FrankaRobosuiteCubesLowLevel
from capx.safety.config import SafetyConfig
from capx.safety.rules import (
    SafetyStopError,
    capsule_to_spheres,
    points_to_capsule_distance,
    speed_limit,
)

HUMAN_BODY = "human_proxy"  # MuJoCo body name of the human proxy
HUMAN_GEOM = "human_proxy_geom"  # MuJoCo geom name of the human proxy


# ------------------------------ robosuite level ------------------------------
class StackWithHuman(Stack):
    """robosuite Stack task + a capsule representing a standing human (visual only)."""

    def __init__(self, human_radius: float = 0.18, human_half_height: float = 0.65, **kwargs):
        self.human_radius = human_radius  # capsule radius
        self.human_half_height = human_half_height  # capsule half length
        self.human_world_pos = None  # set by the capx wrapper (needs the robot base pose)
        super().__init__(**kwargs)

    def _load_model(self):
        """Build the normal Stack scene, then append the human capsule to the world body."""
        super()._load_model()
        body = ET.Element("body", name=HUMAN_BODY, pos="0 1 1")  # temporary pose, moved on reset
        ET.SubElement(
            body,
            "geom",
            name=HUMAN_GEOM,
            type="capsule",
            size=f"{self.human_radius} {self.human_half_height}",
            rgba="0.9 0.6 0.4 0.6",  # skin-like, semi transparent
            contype="0",  # no physical collision: violations are measured by distance
            conaffinity="0",
            group="1",  # visual group, rendered by cameras
        )
        self.model.worldbody.append(body)

    def _reset_internal(self):
        """Called by robosuite on every reset (also after a hard reset rebuilds the sim)."""
        super()._reset_internal()
        self.place_human()

    def place_human(self) -> None:
        """Move/resize the capsule according to the current settings."""
        if self.human_world_pos is None:  # not configured yet (during __init__)
            return
        body_id = self.sim.model.body_name2id(HUMAN_BODY)
        geom_id = self.sim.model.geom_name2id(HUMAN_GEOM)
        self.sim.model.body_pos[body_id] = self.human_world_pos  # body has no joint -> fixed pose
        self.sim.model.geom_size[geom_id, 0] = self.human_radius
        self.sim.model.geom_size[geom_id, 1] = self.human_half_height
        self.sim.forward()  # update derived positions (xpos) and rendering


# -------------------------------- capx level --------------------------------
class FrankaRobosuiteCubesHumanLowLevel(FrankaRobosuiteCubesLowLevel):
    """Cube-stack low-level env with a human proxy and speed & separation monitoring."""

    def __init__(
        self,
        controller_cfg: str = "capx/integrations/robosuite/controllers/config/robots/panda_joint_ctrl.json",
        max_steps: int = 1500,
        seed: int | None = None,
        viser_debug: bool = False,
        privileged: bool = False,
        enable_render: bool = False,
    ) -> None:
        # Skip FrankaRobosuiteCubesLowLevel.__init__ (it hardcodes Stack); init the shared base only
        RobosuiteBaseEnv.__init__(
            self,
            controller_cfg=controller_cfg,
            max_steps=max_steps,
            seed=seed,
            viser_debug=False,
            privileged=privileged,
            enable_render=enable_render,
        )
        # Same settings as the original non-privileged cube env, but headless and with the human
        self.robosuite_env = StackWithHuman(
            robots=["Panda"],
            has_renderer=False,
            has_offscreen_renderer=True,
            camera_names=self.render_camera_names,
            camera_depths=True,
            renderer="mujoco",
            camera_heights=self._render_height,
            camera_widths=self._render_width,
            controller_configs=load_composite_controller_config(controller=self.controller_cfg),
            horizon=max_steps,
            reward_shaping=True,
        )
        # Cube placement identical to FrankaRobosuiteCubesLowLevel
        cubes = [self.robosuite_env.cubeA, self.robosuite_env.cubeB]
        self.robosuite_env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=cubes,
            x_range=[-0.18, 0.18],
            y_range=[-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.robosuite_env.table_offset,
            z_offset=0.01,
            rng=self.robosuite_env.rng,
        )
        self._init_robot_links()
        self._init_viser_debug(False)

        # Robot bodies used for the distance check (arm links + gripper)
        model = self.robosuite_env.sim.model
        self._robot_body_ids = [
            i
            for i in range(model.nbody)
            if (model.body_id2name(i) or "").startswith(("robot0_", "gripper0_"))
        ]

        self._reset_safety_report()
        self.set_safety_config(SafetyConfig())  # defaults; the task env overrides from YAML

    # ---------------------------- configuration ----------------------------
    def set_safety_config(self, cfg: SafetyConfig) -> None:
        """Apply safety settings and place the human (position given in robot base frame)."""
        self.safety = cfg
        base = vtf.SE3(wxyz_xyz=self.base_link_wxyz_xyz)  # robot base pose in world
        pos_base = np.array(list(cfg.human_position), dtype=np.float64)
        human_world = base @ vtf.SE3.from_translation(pos_base)  # base frame -> world frame
        self.robosuite_env.human_radius = float(cfg.human_radius)
        self.robosuite_env.human_half_height = float(cfg.human_half_height)
        self.robosuite_env.human_world_pos = human_world.translation()
        self.robosuite_env.place_human()

    # ------------------------------ human info ------------------------------
    def get_human_capsule(self) -> dict[str, Any]:
        """Human capsule in robot base frame (same frame as get_object_pose)."""
        return {
            "position": np.array(list(self.safety.human_position), dtype=np.float64),
            "radius": float(self.safety.human_radius),
            "half_height": float(self.safety.human_half_height),
        }

    def get_human_obstacles(self) -> list[dict]:
        """Human as spheres (robot base frame) for the collision-aware IK server."""
        h = self.get_human_capsule()
        return capsule_to_spheres(h["position"], h["radius"], h["half_height"])

    def get_min_distance_to_human(self) -> float:
        """Approximate clearance [m] between the robot and the human surface."""
        points = self.robosuite_env.sim.data.xpos[self._robot_body_ids]  # robot body origins (world)
        d = points_to_capsule_distance(
            points,
            self.robosuite_env.human_world_pos,
            self.safety.human_radius,
            self.safety.human_half_height,
        )
        return d - self.safety.robot_link_radius  # links have a thickness

    # ------------------------------ reporting ------------------------------
    def _reset_safety_report(self) -> None:
        self._safety_report = {
            "min_distance": float("inf"),  # smallest clearance seen this episode
            "slow_steps": 0,  # sim steps spent in the slow zone
            "violation_steps": 0,  # sim steps below min_separation
            "stop_events": 0,  # motions aborted by a protective stop
        }

    def _log_distance(self, dist: float) -> None:
        r = self._safety_report
        r["min_distance"] = min(r["min_distance"], dist)
        if dist < self.safety.min_separation:
            r["violation_steps"] += 1
        elif dist < self.safety.slow_zone:
            r["slow_steps"] += 1

    def get_safety_report(self) -> dict[str, Any]:
        return dict(self._safety_report)  # copy so callers cannot modify it

    # ------------------------------ control ------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._reset_safety_report()  # new episode -> new report
        return super().reset(seed=seed, options=options)  # human is re-placed in _reset_internal

    def _measured_joints(self) -> np.ndarray:
        obs = self.robosuite_env._get_observations()
        return np.array(obs["robot0_joint_pos"], dtype=np.float64)

    def _send_joint_command(self, cmd: np.ndarray) -> None:
        """One sim step towards joint setpoint `cmd` (same bookkeeping as the base class)."""
        action = np.concatenate([cmd, [self._gripper_fraction, self._gripper_fraction]])
        action[-2:] = 1.0 - action[-2:] * 2.0  # gripper mapping used by the base class
        self._do_robosuite_step(action)
        if self._record_frames and self._sim_step_count % self._subsample_rate == 0:
            self._record_frame()
        self._sim_step_count += 1

    def move_to_joints_blocking(
        self, joints: np.ndarray, *, tolerance: float = 0.02, max_steps: int = 100
    ) -> None:
        """Move to target joints with an SSM speed limit (overrides the base implementation).

        Instead of sending the final target directly, a setpoint `cmd` advances towards it by at
        most v_max * dt per step, where v_max depends on the robot<->human clearance.
        """
        cfg = self.safety
        target = np.asarray(joints, dtype=np.float64).reshape(7)
        dt = 1.0 / self.robosuite_env.control_freq  # duration of one sim step [s]
        cmd = self._measured_joints()  # setpoint starts at the current joints
        # Slow motions need more steps than the default budget
        budget = max_steps + int(np.ceil(np.max(np.abs(target - cmd)) / (cfg.v_max_reduced * dt)))

        prev_dist = None  # clearance at the previous step
        holds = 0  # number of steps spent in protective stop
        for _ in range(budget):
            current = self._measured_joints()
            if np.linalg.norm(current - target) < tolerance:  # reached
                break

            dist = self.get_min_distance_to_human()
            self._log_distance(dist)
            v = speed_limit(dist, cfg)  # allowed joint speed

            if v == 0.0:  # inside min_separation
                getting_closer = prev_dist is not None and dist < prev_dist - 1e-4
                if getting_closer:  # protective stop: hold the current position
                    cmd = current.copy()
                    holds += 1
                    if holds >= cfg.max_stop_steps:  # stopped too long -> abort motion
                        self._current_joints = cmd
                        self._on_protective_stop(dist)
                        return
                else:  # not getting closer: allow a slow move so the robot can back away
                    v = cfg.v_max_reduced

            if v > 0.0:  # advance the setpoint with the speed limit
                cmd = cmd + np.clip(target - cmd, -v * dt, v * dt)

            prev_dist = dist
            self._send_joint_command(cmd)

        self._current_joints = cmd  # gripper actions (_step_once) keep this arm setpoint

    def _on_protective_stop(self, dist: float) -> None:
        self._safety_report["stop_events"] += 1
        msg = (
            f"Protective stop: robot came within {dist:.3f} m of the human "
            f"(min_separation={self.safety.min_separation} m). Motion aborted."
        )
        print(f"[SAFETY] {msg}")
        if self.safety.raise_on_stop:  # stop the generated program so the LLM can replan
            raise SafetyStopError(msg)


__all__ = ["StackWithHuman", "FrankaRobosuiteCubesHumanLowLevel"]
