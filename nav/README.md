# Spot with arm — navigation (Nav2) part

Spot with arm walks through a maze with **ROS 2 Nav2**, picks a cube found by its **hand camera**, carries it through a second maze
and places it in a bin. Simulation only: NVIDIA Isaac Sim 6.0 / Isaac Lab 3.0, ROS 2 Humble.

## Run
```bash
@SPOT_DEMO_ROOT@/build_demo.sh                        # once, and after any change to full_demo_perc.py or the patch scripts
@SPOT_DEMO_ROOT@/run_demo.sh                          # default scene; press g when it says Nav2 is ready
@SPOT_DEMO_ROOT@/run_demo.sh --scene bench_north_east # bench moved + turned, bin elsewhere
@SPOT_DEMO_ROOT@/run_demo.sh --amcl                   # drifting odometry + AMCL localisation
@SPOT_DEMO_ROOT@/run_demo.sh --no_mover               # without the moving obstacle
```
Logs: Isaac `/tmp/demo_run.log`, ROS stack `/tmp/nav_up.log` and `/tmp/nav_logs/`. Success line: `[RESULT] CUBE IN BIN`.
Check a scene without Isaac (1 s): `python3 @SPOT_DEMO_ROOT@/nav/check_scene.py` (`NAV_SCENE=<file>` for another file).

## Operator panel
Window "Spot Nav2 panel" in the Isaac viewer: Start full demo (= g), Go to bench / bin (stand poses, no pick), Go to x, y, yaw (map frame, as in RViz; rejected if < 0.65 m from obstacles), Cancel. Go buttons work when idle (waiting, finished, after a trip). Status: phase, Nav2 status + distance left, Spot pose, last message.

## How it works
rclpy cannot be imported inside Isaac Lab's Python 3.12, hence the small UDP relay.

Phases: FOLD -> **NAV_PICK** (Nav2 to the pre-pose) -> APPROACH (scripted last 0.5 m) -> UNFOLD -> LOOK (hand camera) -> PICK (RL policy v9g)
-> STOW (+ camera check the cube is still held) -> BACKOFF -> **NAV_BIN** -> GO_BIN (scripted last 0.5 m) -> PLACE -> RELEASE -> RETRACT.

## What is sensed and what is given
| | How |
|---|---|
| Obstacles | 2D lidar: 360 PhysX ray casts against the real scene colliders (`phys_scan.py`), z = 0.12 m. Sees objects that are not on the map (orange crate). |
| Moving obstacle | Red kinematic box slides into the route; added to the scan from its live pose, because GPU PhysX ray casts do not update moved bodies (simulation shortcut). |
| Cube position | Hand camera (depth + segmentation), ~1 mm error. |
| Map | Given: generated from `nav_world.py` / `scene.json` (no SLAM). |
| Robot pose | Exact from the simulator by default; `--amcl`: drifting odometry corrected by AMCL (2-12 cm typical, up to ~24 cm after big turns; tuning open). |
| Where to stand / approach side | Computed by `goto_planner.py` from the target and the map (checks room to turn, rejects unsafe sides). |

## Files (`@SPOT_DEMO_ROOT@/nav/`)
`nav_world.py` boxes + start (+ `scene.json` overrides) · `scenes/*.json` presets · `gen_map.py` map · `goto_planner.py` stand/pre-pose planner ·
`check_scene.py` scene check · `nav_link.py` Isaac-side link (StateLink, HeadingHold, GoalClient, Scan2D, OdomDrift) · `phys_scan.py` PhysX lidar ·
`spot_ros_relay.py` · `goal_bridge.py` · `nav_up.sh` ROS stack (`--amcl`) · `config/nav2_params.yaml` · `make_amcl_params.py` · `loc_err.py` localisation error.
Build chain (`@SPOT_DEMO_ROOT@/`): `make_perc_nav.py` -> `make_nav_patch.py` -> `make_perc_nav_scan.py` -> `fix_scan.py` -> `make_amcl_patch.py` -> `make_goto_patch.py`
-> `make_mover_patch.py` -> `fix_mover_scan.py` -> `fix_mover2.py`, all run by `build_demo.sh`. `full_demo_perc.py` (perception) is never edited.

## Tuning that matters (do not undo)
- Walking policy turns at ~-0.10 rad/s when told to stand: +0.10 rad/s bias on every leg command + heading hold at zero command.
- It does not walk below ~0.1 m/s: Nav2 `min_approach_linear_velocity` 0.15.
- Nav2 goal tolerance 0.12 m, yaw any (3.14): precise pose comes from the scripted final approach. A tight Nav2 yaw tolerance loops forever (turning slides the base).
- RPP `max_angular_accel` 6.0. Pre-poses need >= 0.65 m from the base centre to any obstacle to turn in place (enforced by goto_planner).
- Isaac Lab 3.0: PhysX ray casts need `enable_scene_query_support` (on `cfg.sim` / `cfg.sim.physics`).
- Start Isaac before the ROS stack (run_demo.sh does this); restarting Isaac needs a ROS restart (sim clock goes back to 0).

## Known issues
- PLACE: wrist `arm0_wr0` sometimes stalls at full torque, recovers after a slower retry (manipulation side).
- Cube can slip during STOW: the after-STOW camera check retries the pick at the bench.
- Sim runs ~1/6-1/10 real time while GPU training runs; a full demo is ~10-13 min wall time.
