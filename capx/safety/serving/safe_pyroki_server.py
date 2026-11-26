""" A Wrapper around capx/serving/launch_pyroki_server.py that adds collision-aware IK and motion planning.

Reuses the robot loading and obstacle parsing of capx/serving/launch_pyroki_server.py
and runs as a separate server (default port 8117), so the original server is unchanged.

    POST /ik   {target_pose_wxyz_xyz, prev_cfg?, obstacles?} -> {joint_positions}
    POST /plan {start_pose_wxyz_xyz, end_pose_wxyz_xyz, obstacles?, timesteps, dt} -> {waypoints, dt}
"""

from __future__ import annotations

import logging

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import capx.integrations.motion.pyroki_snippets as pks
import capx.serving.launch_pyroki_server as lps  # original server: robot globals + helpers

logger = logging.getLogger("safe_pyroki_server")
app = FastAPI()


class SafeIkRequest(BaseModel):
    target_pose_wxyz_xyz: list[float]  # wxyz + xyz (length 7)
    prev_cfg: list[float] | None = None
    obstacles: list[dict] | None = None  # same format as launch_pyroki_server._build_world_coll


def _do_safe_ik(req: SafeIkRequest) -> list[float]:
    pose = np.array(req.target_pose_wxyz_xyz, dtype=np.float64)
    prev = np.array(req.prev_cfg, dtype=np.float64) if req.prev_cfg is not None else None
    if not req.obstacles:  # no obstacles -> exact same behaviour as the original server
        return lps._do_solve_ik(pose, prev)
    world = lps._build_world_coll(req.obstacles)  # dicts -> PyRoKi collision geometries
    q = pks.solve_ik_with_collision(  # note: ignores prev_cfg (no velocity cost)
        robot=lps._ROBOT,
        coll=lps._ROBOT_COLL,
        world_coll_list=world,
        target_link_name=lps._TARGET_LINK,
        target_position=pose[4:],
        target_wxyz=pose[:4],
    )
    return list(map(float, q))


def _do_safe_plan(req: lps.PlanRequest) -> lps.PlanResponse:
    if not req.obstacles:  # no obstacles -> original linear-IK planner
        return lps._do_plan_motion(req)
    start = np.array(req.start_pose_wxyz_xyz, dtype=np.float64)
    end = np.array(req.end_pose_wxyz_xyz, dtype=np.float64)
    traj = pks.solve_trajopt(  # trajectory optimisation with swept-volume collision cost
        robot=lps._ROBOT,
        robot_coll=lps._ROBOT_COLL,
        world_coll=lps._build_world_coll(req.obstacles),
        target_link_name=lps._TARGET_LINK,
        start_position=start[4:],
        start_wxyz=start[:4],
        end_position=end[4:],
        end_wxyz=end[:4],
        timesteps=req.timesteps,
        dt=req.dt,
    )
    return lps.PlanResponse(waypoints=np.asarray(traj).tolist(), dt=float(req.dt))


@app.post("/ik", response_model=lps.IkResponse)
async def safe_ik(req: SafeIkRequest):
    if lps._ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")
    try:
        joints = await lps._run_in_thread(_do_safe_ik, req)  # keep the event loop free
    except Exception as e:
        logger.exception("Safe IK failed")
        raise HTTPException(500, f"Safe IK solve failed: {e}")
    return lps.IkResponse(joint_positions=joints)


@app.post("/plan", response_model=lps.PlanResponse)
async def safe_plan(req: lps.PlanRequest):
    if lps._ROBOT is None:
        raise HTTPException(503, "Pyroki not initialized")
    try:
        return await lps._run_in_thread(_do_safe_plan, req)
    except Exception as e:
        logger.exception("Safe planning failed")
        raise HTTPException(500, f"Safe motion planning failed: {e}")


def _warm_up() -> None:
    """Run one collision IK so JAX compiles before the first real request."""
    try:
        from capx.safety.rules import capsule_to_spheres  # pure numpy, no robosuite import

        req = SafeIkRequest(
            target_pose_wxyz_xyz=[0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.3],  # gripper down, in front
            obstacles=capsule_to_spheres([0.56, 0.55, -0.06], 0.18, 0.65),  # default human
        )
        _do_safe_ik(req)
        logger.info("Safe IK warm-up done")
    except Exception:
        logger.exception("Safe IK warm-up failed (server still starts)")


def main(
    robot: str = "panda_description",
    target_link: str = "panda_hand",
    port: int = 8117,  #run on a separte port from the pyroki original server (8116)
    host: str = "127.0.0.1",
):
    # loads robot once and keeps it in memory for all future requests
    lps.init_pyroki_server(robot_urdf_name=robot, target_link_name=target_link)  
    
    # warm up JAX compilation for the first request moves that initial delay to startup 
    # rather than making the first real client request unusually slow.
    _warm_up() 

    # run the HTTP server
    # uvcorn listens to requested port and host, and forwards requests to FastAPI app
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
