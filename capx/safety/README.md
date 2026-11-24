# capx.safety — human-aware collision avoidance

Adds a **restriction layer** to the CaP-X flow for scenes where a human is present.
Everything lives in this folder + one YAML; no existing CaP-X file is modified.

## Layers

| Layer | Where | What |
|---|---|---|
| High level (prompt) | `tasks/cube_stack_human.py` | `restrictions` from YAML are added to the LLM prompt (between task and API docs). |
| Planning | `api/human_aware_control.py` + `serving/safe_pyroki_server.py` | `goto_pose` sends the human (as spheres) to a collision-aware IK (`solve_ik_with_collision`). |
| Low level (control) | `sim/robosuite_cubes_human.py` | `move_to_joints_blocking` limits joint speed by distance (SSM) and stops when too close. |
| Pipeline feedback | `tasks/cube_stack_human.py` | Safety report in `info["safety"]`; violations appended to `stderr` (seen by multi-turn LLM). |

## Flow

```
YAML (restrictions + safety)
  -> HumanAwareCubeStackCodeEnv  (prompt += restrictions, set_safety_config -> place human)
  -> LLM code calls goto_pose()
       -> HumanAwareControlApi._safe_ik_solve -> :8117 /ik with obstacles
       -> FrankaRobosuiteCubesHumanLowLevel.move_to_joints_blocking  (SSM speed limit / stop)
  -> step() adds safety report to info / stderr
```

## Files

- `config.py` — `SafetyConfig` (numbers), `SafeCodeExecEnvConfig` (adds `restrictions`, `safety`).
- `rules.py` — distance to capsule, capsule→spheres, `speed_limit()` (SSM rule), `SafetyStopError`.
- `sim/robosuite_cubes_human.py` — `StackWithHuman` (robosuite) + `FrankaRobosuiteCubesHumanLowLevel` (capx).
- `api/human_aware_control.py` — `HumanAwareControlApi`: `get_human_pose()`, `get_min_distance_to_human()`.
- `tasks/cube_stack_human.py` — task env + registration of env `franka_robosuite_cubes_human_low_level` and API `HumanAwareControlApi`.
- `serving/safe_pyroki_server.py` — `/ik` and `/plan` accepting `obstacles` (port 8117).
- `env_configs/safety/franka_robosuite_cube_stack_human.yaml` — run config.

## SSM rule (speed & separation monitoring, ISO/TS 15066 idea)

```
clearance <  min_separation  -> stop (hold); abort after max_stop_steps -> SafetyStopError
clearance <  slow_zone       -> v_max_reduced
otherwise                    -> v_max_nominal
```
Clearance = distance from robot body origins to the human capsule surface − `robot_link_radius`.
Inside `min_separation` the robot may still move slowly if it is not getting closer (lets it back away).

## Run

```bash
# Oracle code first (checks env, human placement, IK server)
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/safety/franka_robosuite_cube_stack_human.yaml \
    --use-oracle-code True --total-trials 2

# With an LLM
uv run --no-sync --active capx/envs/launch.py \
    --config-path env_configs/safety/franka_robosuite_cube_stack_human.yaml
```

## Known limitations (first version)

- Human is **static** and **privileged** (pose from config, not perception).
- Human is **visual only** (no physical contact); violations are measured by distance.
- Distance uses robot body origins + a fixed link radius (approximation).
- Collision IK only keeps the **final** configuration away from the human; the joint-space path
  between configurations is protected only by the SSM speed limit / stop.
- `solve_ik_with_collision` ignores `prev_cfg`, so consecutive IK solutions may jump more.
- The safety report is cumulative over the episode.

## Next steps

1. Moving human: make the capsule a mocap body and update `sim.data.mocap_pos` each step.
2. Perception: SAM3 "person"/"hand" + depth → points → spheres instead of config pose.
3. Use `/plan` (trajopt with swept collision) for long motions.
4. Static check of generated code (e.g. forbid direct `env.` calls) before execution.
