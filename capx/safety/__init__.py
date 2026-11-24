"""Human-aware collision avoidance for CaP-X (restrictions + speed & separation monitoring).

Kept intentionally empty: importing this package must stay lightweight because the
safe PyRoKi server (capx.safety.serving) also lives here and should not import robosuite.
Registration of the env + API happens in capx/safety/tasks/cube_stack_human.py.
See capx/safety/README.md.
"""
