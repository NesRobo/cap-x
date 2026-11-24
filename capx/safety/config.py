"""Configuration for the human-aware safety layer."""

from __future__ import annotations

from dataclasses import dataclass, field

from capx.envs.tasks.base import CodeExecEnvConfig

# High-level restrictions shown to the LLM when the YAML does not provide any
DEFAULT_RESTRICTIONS: list[str] = [
    "A human is present next to the table. Never move the robot toward the human.",
    "Keep every part of the robot at least the minimum separation away from the human.",
    "Never carry an object over or near the human; route lateral motions on the side away from the human.",
    "Prefer lifting objects straight up before moving laterally, and keep motions short near the human.",
]


@dataclass
class SafetyConfig:
    """Numbers used by the low-level safety layer (all positions in robot base frame, meters)."""

    # Human proxy = vertical capsule (standing person)
    human_position: list[float] = field(default_factory=lambda: [0.56, 0.55, -0.06])  # capsule center
    human_radius: float = 0.18  # capsule radius (body width)
    human_half_height: float = 0.65  # half length of the capsule axis

    # Speed & separation monitoring (SSM) thresholds
    min_separation: float = 0.05  # below this clearance -> protective stop
    slow_zone: float = 0.25  # below this clearance -> reduced speed
    v_max_nominal: float = 1.5  # max joint speed far from the human [rad/s]
    v_max_reduced: float = 0.3  # max joint speed inside the slow zone [rad/s]

    # Approximation / behaviour
    robot_link_radius: float = 0.08  # robot links are approximated as spheres of this radius
    max_stop_steps: int = 20  # sim steps the robot may stay stopped before aborting the motion
    raise_on_stop: bool = True  # raise SafetyStopError when a motion is aborted
    fail_on_violation: bool = False  # if True, any violation sets the reward to 0


@dataclass
class SafeCodeExecEnvConfig(CodeExecEnvConfig):
    """CodeExecEnvConfig + high-level restrictions + low-level safety numbers."""

    restrictions: list[str] | None = None  # high-level rules added to the LLM prompt
    safety: SafetyConfig = field(default_factory=SafetyConfig)  # low-level enforcement settings
