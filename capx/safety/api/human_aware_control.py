"""FrankaControlApi variant that is aware of the human (collision-aware IK + human queries)."""

from __future__ import annotations

from typing import Any

import numpy as np

from capx.envs.base import BaseEnv
from capx.integrations.franka.control import FrankaControlApi
from capx.utils.serve_utils import post_with_retries

SAFE_PYROKI_URL = "http://127.0.0.1:8117"  # capx/safety/serving/safe_pyroki_server.py


class HumanAwareControlApi(FrankaControlApi):
    """Same functions as FrankaControlApi, plus human queries; goto_pose avoids the human."""

    def __init__(self, env: BaseEnv, safe_server_url: str = SAFE_PYROKI_URL, **kwargs: Any) -> None:
        super().__init__(env, **kwargs)
        self._safe_server_url = safe_server_url.rstrip("/")
        # The parent goto_pose() calls self.ik_solve_fn -> swap it for the collision-aware one
        self.ik_solve_fn = self._safe_ik_solve

    def _safe_ik_solve(
        self, target_pose_wxyz_xyz: np.ndarray, prev_cfg: np.ndarray | None = None
    ) -> np.ndarray:
        """IK request that also sends the human as obstacles (same signature as init_pyroki)."""
        payload = {
            "target_pose_wxyz_xyz": np.asarray(target_pose_wxyz_xyz).tolist(),
            "prev_cfg": np.asarray(prev_cfg).tolist() if prev_cfg is not None else None,
            "obstacles": self._env.get_human_obstacles(),  # spheres in robot base frame
        }
        # Long timeout: the first collision IK call triggers JAX compilation on the server
        data = post_with_retries(f"{self._safe_server_url}/ik", payload, timeout_seconds=120.0)
        return np.asarray(data["joint_positions"], dtype=np.float32)

    def functions(self) -> dict[str, Any]:
        fns = super().functions()
        fns["get_human_pose"] = self.get_human_pose
        fns["get_min_distance_to_human"] = self.get_min_distance_to_human
        return fns

    def get_human_pose(self) -> tuple[np.ndarray, float, float]:
        """Get the region occupied by the human, modelled as a vertical capsule.
        Uses the same frame as get_object_pose. Plan motions that stay clear of this region.

        Args:
            None

        Returns:
            position: (3,) XYZ center of the capsule in meters.
            radius: (float) capsule radius in meters.
            half_height: (float) half length of the capsule vertical axis in meters.
        """
        h = self._env.get_human_capsule()
        self._log_step("get_human_pose", f"Human at {np.array2string(h['position'], precision=3)}")
        return h["position"], h["radius"], h["half_height"]

    def get_min_distance_to_human(self) -> float:
        """Get the current approximate clearance between the robot and the human.

        Args:
            None

        Returns:
            distance: (float) clearance in meters. The robot slows down when this is small
                and stops (SafetyStopError) when it is below the minimum separation.
        """
        return float(self._env.get_min_distance_to_human())
