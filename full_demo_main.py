"""Full demo with Nav2 + hand-camera perception:
FOLD -> NAV_PICK (Nav2 to pre-pick) -> APPROACH (scripted last 0.5 m) -> UNFOLD -> SETTLE -> LOOK -> PICK (camera, verify, retry)
-> STOW -> BACKOFF (0.6 m) -> NAV_BIN (Nav2 to pre-place) -> GO_BIN (scripted last 0.5 m) -> PLACE -> RELEASE -> RETRACT.
Needs @SPOT_DEMO_ROOT@/nav/nav_up.sh running. Room, bench, bin and start come from @SPOT_DEMO_ROOT@/nav/nav_world.py. --no_nav = scripted walk.
Press g in THIS terminal to start (start your screen recording first)."""
import argparse, math, sys, select, termios, tty, threading, time
T_START = time.time()
sys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, help="exported v5 policy.pt")
parser.add_argument("--shift", type=float, default=1.5, help="bench distance added in front of the start (m)")
parser.add_argument("--bin_x", type=float, default=None, help="bin x (default: nav_world.BIN)")
parser.add_argument("--bin_y", type=float, default=None, help="bin y (default: nav_world.BIN)")
parser.add_argument("--start_x", type=float, default=None, help="robot start x (default: nav_world.START)")
parser.add_argument("--start_y", type=float, default=None, help="robot start y (default: nav_world.START)")
parser.add_argument("--no_nav", action="store_true", help="NAV: scripted straight-line walk instead of Nav2")
parser.add_argument("--pre_dist", type=float, default=0.5, help="NAV: Nav2 stops this far before the pick/place stand pose (m)")
parser.add_argument("--nav_timeout", type=float, default=150.0, help="NAV: sim seconds per Nav2 goal before the scripted fallback")
parser.add_argument("--wz_bias", type=float, default=0.10, help="NAV: walking-policy yaw-rate offset added to every leg command")
parser.add_argument("--no_script_bias", action="store_true", help="NAV: do not add wz_bias to the scripted (non-Nav2) leg commands")
parser.add_argument("--scan", choices=["physx", "boxes"], default="physx", help="SCAN: physx = ray casts into the real scene, boxes = old nav_world list")
parser.add_argument("--scan_rays", type=int, default=360, help="SCAN: rays per 360 deg scan")
parser.add_argument("--scan_h", type=float, default=0.12, help="SCAN: height of the scan plane above the floor (m); below the bin top 0.146")
parser.add_argument("--no_surprise", action="store_true", help="SCAN: do not spawn the surprise obstacle (not in the Nav2 map)")
parser.add_argument("--scan_test", action="store_true", help="SCAN: compare physx scan vs box scan at the start pose, then quit")
parser.add_argument("--no_drift", action="store_true", help="ODOM: send exact odometry (no drift)")
parser.add_argument("--no_mover", action="store_true", help="MOVER: no moving obstacle")
parser.add_argument("--cube_slot", type=int, default=0, choices=[-1, 0, 1], help="MULTI: task cube slot along the bench (-1, 0, +1; spacing 0.30 m)")
parser.add_argument("--distractors", action="store_true", help="MULTI: coloured neighbour cubes in the other slots")
parser.add_argument("--slot_gap", type=float, default=0.30, help="MULTI: distance between cube slots (m)")
parser.add_argument("--multi_flow", action="store_true", help="FLOW: ask which cube, pick it, bin it, repeat until none left (implies --distractors)")
parser.add_argument("--cam_detect", action="store_true", help="FLOW: list the cubes with the hand camera (detect_cubes) instead of the sim")
parser.add_argument("--scan_pan", type=float, default=0.5, help="FLOW: shoulder pan (rad) for the left/right camera scan poses")
parser.add_argument("--body_cam", action="store_true", help="FLOW: identify the cubes with a front BODY camera (wrist camera still picks)")
parser.add_argument("--body_cam_link", type=str, default="base", help="BODY CAM: robot link the camera is mounted on")
parser.add_argument("--body_cam_pos", type=float, nargs=3, default=[0.47, 0.0, 0.0], help="BODY CAM: mount position in the link frame (m)")
parser.add_argument("--body_cam_pitch", type=float, default=20.0, help="BODY CAM: tilt down (deg)")
parser.add_argument("--body_cam_hfov", type=float, default=90.0, help="BODY CAM: horizontal field of view (deg)")
parser.add_argument("--stand_extra", type=float, default=0.03, help="FLOW: plan the cube stand pose this much further back (m)")
parser.add_argument("--order", type=str, default="", help="FLOW: automatic answers, e.g. red,dex,blue (otherwise panel buttons / typing)")
parser.add_argument("--mover_test", action="store_true", help="MOVER: slide the box once with Spot standing, report, quit")
parser.add_argument("--odom_scale", type=float, default=0.03, help="ODOM: speed scale error (0.03 = odometry reads 3%% long)")
parser.add_argument("--odom_yaw_bias", type=float, default=0.003, help="ODOM: gyro yaw-rate bias (rad/s); 0.003 = 0.17 deg/s")
parser.add_argument("--odom_noise", type=float, default=0.02, help="ODOM: velocity noise, fraction of speed")
parser.add_argument("--place_dist", type=float, default=0.88, help="body-centre to bin-centre distance when placing (m)")
parser.add_argument("--bin_scale", type=float, default=1.6, help="bin width/length scale (height unchanged)")
parser.add_argument("--arm_vel", type=float, default=None, help="TEST: override arm joint velocity limit (rad/s)")
parser.add_argument("--arm_kp", type=float, default=None, help="TEST: arm stiffness")
parser.add_argument("--arm_kd", type=float, default=None, help="TEST: arm damping")
parser.add_argument("--fast_kp", type=float, default=None, help="arm stiffness for SCRIPTED moves (PICK keeps training gains)")
parser.add_argument("--fast_kd", type=float, default=120.0, help="arm damping for scripted moves")
parser.add_argument("--move_time", type=float, default=2.5, help="seconds per scripted arm move")
parser.add_argument("--task", default="Isaac-Lift-Cube-SpotArm-Stand-v5-Play", help="Play task matching the --arm policy")
parser.add_argument("--auto", action="store_true", help="start immediately instead of waiting for g")
parser.add_argument("--look_s", type=float, default=0.3, help="PERC: seconds of hand-camera frames for the cube estimate")
parser.add_argument("--retries", type=int, default=2, help="PERC: pick retries after a failed pick")
parser.add_argument("--stall_steps", type=int, default=2, help="PERC: finger still this many steps = grasp")
parser.add_argument("--gofs", type=float, nargs=3, default=[0.227, -0.015, -0.007], help="PERC: grasped cube centre in wr1 frame (m)")
parser.add_argument("--lift_check", type=float, default=0.20, help="PERC: wrist rise after grasp before the camera check (m)")
parser.add_argument("--inhand_px", type=int, default=20000, help="PERC: cube pixels needed to count as in hand")
parser.add_argument("--inhand_d", type=float, default=0.17, help="PERC: max median cube depth to count as in hand (m)")
parser.add_argument("--pick_timeout", type=float, default=12.0, help="PERC: seconds per pick attempt")
parser.add_argument("--const_gofs", action="store_true", help="PERC: use the fixed grasp offset instead of per-grasp")
parser.add_argument("--exit_on_done", action="store_true", help="quit 1 s after DONE (for batch runs)")
parser.add_argument("--pick_x", type=float, default=0.875, help="PERC: wanted cube distance ahead of the base (m)")
parser.add_argument("--pick_tol_x", type=float, default=0.07, help="PERC: re-park if forward error is larger (m)")
parser.add_argument("--pick_tol_y", type=float, default=0.07, help="PERC: re-park if sideways error is larger (m)")
parser.add_argument("--carry_slow", type=float, default=2.0, help="PERC: scripted moves with a closed gripper are this much slower")
parser.add_argument("--look_min", type=int, default=8, help="PERC: good frames wanted for a LOOK")
parser.add_argument("--look_min_late", type=int, default=5, help="PERC: accepted after 1 s if look_min not reached")
parser.add_argument("--place_tol", type=float, default=0.05, help="PERC: carried-cube offset from bin centre that triggers one fix step (m)")
parser.add_argument("--place_fix_max", type=float, default=0.10, help="PERC: max base step at the bin (m)")
parser.add_argument("--wrist_kp", type=float, default=None, help="wrist stiffness for SCRIPTED moves (default: training gains)")
parser.add_argument("--wrist_kd", type=float, default=60.0, help="wrist damping for scripted moves when --wrist_kp is set")
parser.add_argument("--neighbours", action="store_true", help="TEST: add red/blue cubes 0.30 m left/right of the task cube")
parser.add_argument("--detect_test", action="store_true", help="TEST: print detect_cubes() at each first LOOK")
parser.add_argument("--repark_vmin", type=float, default=0.22, help="PERC: minimum walking speed for re-park / place fix (m/s)")
parser.add_argument("--stow_el0", type=float, default=2.8, help="elbow angle of the stow/carry pose (was 3.1)")
parser.add_argument("--wrist_effort", type=float, default=None, help="TEST: wrist torque limit (N*m) for scripted moves")
parser.add_argument("--grasp_check_s", type=float, default=1.0, help="PERC: seconds after grasp detection before the hand-camera check")
parser.add_argument("--max_yaw_deg", type=float, default=15.0, help="PERC: skip the cube if it is rotated more than this (deg)")
parser.add_argument("--max_tilt_deg", type=float, default=15.0, help="PERC: skip the cube if its top face is tilted more than this (deg)")
parser.add_argument("--sh0_sign", type=float, default=1.0, help="PERC: +1 if positive sh0 turns the arm left (flip if PLACE sideways fix goes the wrong way)")
parser.add_argument("--place_wrist_tol", type=float, default=0.5, help="PERC: PLACE moves done if only wr0 is off by less than this (rad)")
parser.add_argument("--no_gate", dest="gate", action="store_false", help="PERC: disable the grasp gate")
parser.add_argument("--gate_y0", type=float, default=-0.008, help="PERC: usual sideways cube offset in the wrist frame at a good grasp (m)")
parser.add_argument("--gate_y", type=float, default=0.022, help="PERC: allowed sideways deviation before closing (m)")
parser.add_argument("--gate_max_s", type=float, default=4.0, help="PERC: vetoed closing this long -> retry with a fresh LOOK")
parser.add_argument("--hold_wz_min", type=float, default=0.30, help="minimum heading-hold turn rate outside PICK (rad/s)")
parser.add_argument("--park_back", type=float, default=0.0, help="stand this much further back from the bench (m)")
parser.add_argument("--gate_z0", type=float, default=0.005, help="PERC: usual closing-axis cube offset in the wrist frame (m)")
parser.add_argument("--gate_z", type=float, default=0.030, help="PERC: allowed closing-axis deviation before closing (m)")
parser.add_argument("--grip_kp", type=float, default=1500.0, help="finger stiffness in scripted phases (firmer hold while carrying); training value in PICK")
parser.add_argument("--pick_legs_zero", action="store_true", help="TEST: legs command exactly 0 during PICK (as in training)")
parser.add_argument("--settle_train_gains", action="store_true", help="TEST: training arm gains from SETTLE/LOOK on")
parser.add_argument("--settle_tight", action="store_true", help="TEST: SETTLE waits for arm err < settle_arm_tol and vel < 0.05")
parser.add_argument("--settle_arm_tol", type=float, default=0.05, help="arm error for --settle_tight (rad)")
parser.add_argument("--gate", dest="gate", action="store_true", help="PERC: re-enable the grasp gate (off by default since v8.24)")
parser.set_defaults(pick_legs_zero=True, settle_train_gains=True, settle_tight=True, gate=False)   # v8.24: runtime fix on, gate off
parser.add_argument("--cam_pos", type=float, nargs=3, default=[0.10, 0.0, 0.0], help="PERC: camera offset in wr1 frame (m)")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
if args.multi_flow:
    args.distractors = True
args.enable_cameras = True
app = AppLauncher(args).app

import torch
import numpy as np
import gymnasium as gym
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat, subtract_frame_transforms, quat_apply, quat_conjugate
import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.manager_based.manipulation.lift.config.spot_arm.env_cfg import BENCH_X, BENCH_H, CUBE_Z, LIFTED_Z
import nav_world as W
from nav_link import StateLink, HeadingHold, GoalClient, Scan2D, OdomDrift

def T(x):
    if torch.is_tensor(x):
        return x
    import warp as wp
    return wp.to_torch(x)

STOW = {"arm0_sh0": 0.0, "arm0_sh1": -3.1, "arm0_el0": args.stow_el0, "arm0_el1": 0.0, "arm0_wr0": 0.0, "arm0_wr1": 0.0}
PLACE = {"arm0_sh0": 0.0, "arm0_sh1": -0.9, "arm0_el0": 1.8, "arm0_el1": 0.0, "arm0_wr0": 0.6, "arm0_wr1": 0.0}
ARM_SCALE = 0.5
POS_TOL, YAW_TOL = 0.08, 0.10           # parking tolerance: 8 cm, ~6 deg
ARM_TOL = 0.12                           # arm move done when every joint within 0.1 rad
ARM_MAX_WAIT = 15.0                      # seconds allowed per arm move
BIN_H = 0.146
ARM_TOL_NEAR = 0.25                      # close enough when no longer improving
ARM_STUCK_S = 2.0                        # no progress this long after interpolation = stuck / settled

task = args.task
print(f"[DEMO] task: {task}", flush=True)
cfg = parse_env_cfg(task, device=args.device, num_envs=1)
cfg.episode_length_s = 1.0e6
if args.scan == "physx":                       # PhysX ray casts need scene queries; Isaac Lab disables them by default
    _sq = []
    if hasattr(cfg.sim, "enable_scene_query_support"):
        cfg.sim.enable_scene_query_support = True; _sq.append("sim")
    for _k in dir(cfg.sim):                      # Isaac Lab 3.0: physics settings moved (no cfg.sim.physx)
        if _k.startswith("_"):
            continue
        _v = getattr(cfg.sim, _k, None)
        if _v is not None and not isinstance(_v, (int, float, str, bool, tuple, list)) and hasattr(_v, "enable_scene_query_support"):
            setattr(_v, "enable_scene_query_support", True); _sq.append(f"sim.{_k}")
    if _sq:
        print(f"[SCAN] PhysX scene query support: ON ({_sq})", flush=True)
    else:
        print(f"[SCAN] WARNING: no enable_scene_query_support found. cfg.sim fields: {[k for k in dir(cfg.sim) if not k.startswith('_')]}", flush=True)
cfg.terminations.object_dropping = None
cfg.terminations.base_tilted = None
cfg.terminations.base_low = None
if getattr(cfg.terminations, 'unstable', None) is not None:
    cfg.terminations.unstable = None      # top-down tasks: never auto-reset during the demo
if getattr(cfg.rewards, 'fall_penalty', None) is not None:
    cfg.rewards.fall_penalty = None       # it refers to the fall endings switched off above
cfg.commands.object_pose.resampling_time_range = (1.0e6, 1.0e6)
if args.arm_kp is not None:
    cfg.scene.robot.actuators["spot_arm"].stiffness = args.arm_kp
if args.arm_kd is not None:
    cfg.scene.robot.actuators["spot_arm"].damping = args.arm_kd
if args.arm_kp is not None or args.arm_kd is not None:
    print(f"[DEMO] TEST arm kp={args.arm_kp} kd={args.arm_kd}", flush=True)
if args.arm_vel is not None:
    cfg.scene.robot.actuators["spot_arm"].velocity_limit_sim = args.arm_vel
    print(f"[DEMO] TEST arm velocity limit = {args.arm_vel} rad/s", flush=True)
_bax = (1.0, 0.0) if W.BENCH[3] > W.BENCH[4] else (0.0, 1.0)            # MULTI: bench long axis
def _slot_xy(k):
    return (W.BENCH[1] + _bax[0] * args.slot_gap * k, W.BENCH[2] + _bax[1] * args.slot_gap * k)
CUBE_XY = _slot_xy(args.cube_slot)
cfg.scene.table.init_state.pos = (W.BENCH[1], W.BENCH[2], BENCH_H / 2)          # GOTO: bench from nav_world / scene.json
cfg.scene.object.init_state.pos = [CUBE_XY[0], CUBE_XY[1], CUBE_Z]
print(f"[MULTI] task cube in slot {args.cube_slot:+d} at ({CUBE_XY[0]:+.2f},{CUBE_XY[1]:+.2f}) | task cube spawn: {type(cfg.scene.object.spawn).__name__} "
      f"{getattr(cfg.scene.object.spawn, 'usd_path', '')} scale {getattr(cfg.scene.object.spawn, 'scale', None)}", flush=True)
DISTRACT = {}
if args.distractors:                          # coloured neighbour cubes (dynamic: they can be knocked over -> reported at the end)
    from isaaclab.assets import RigidObjectCfg
    _cols = {"red": (0.85, 0.08, 0.08), "blue": (0.10, 0.25, 0.90), "green": (0.10, 0.70, 0.20)}
    _free = [k for k in (-1, 0, 1) if k != args.cube_slot]
    for _k, (_cn, _rgb) in zip(_free, list(_cols.items())):
        _x, _y = _slot_xy(_k)
        setattr(cfg.scene, f"cube_{_cn}", RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube_" + _cn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(_x, _y, CUBE_Z)),
            spawn=sim_utils.CuboidCfg(size=(0.048, 0.048, 0.048),
                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                      mass_props=sim_utils.MassPropertiesCfg(mass=0.10),
                                      collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_rgb))))
        DISTRACT[_cn] = (_x, _y)
    print(f"[MULTI] neighbour cubes: {DISTRACT}", flush=True)
_long_x = W.BENCH[3] > W.BENCH[4]
if _long_x:                                   # bench turned 90 deg: swap its size instead of rotating it
    _sz = tuple(cfg.scene.table.spawn.size)
    cfg.scene.table.spawn.size = (_sz[1], _sz[0], _sz[2])
print(f"[GOTO] bench at ({W.BENCH[1]:+.2f},{W.BENCH[2]:+.2f}) long side along {'x' if _long_x else 'y'} | scene file {W.SCENE_FILE if W.SCENE else 'none (defaults)'}", flush=True)
cfg.events.reset_object_position.params = {**cfg.events.reset_object_position.params,
                                           "pose_range": ({"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)} if args.distractors else   # MULTI: no big jitter next to neighbours
                                                         {"x": (-0.10, 0.10), "y": (-0.03, 0.03), "z": (0.0, 0.0)} if W.BENCH[3] > W.BENCH[4] else
                                                         {"x": (-0.03, 0.03), "y": (-0.10, 0.10), "z": (0.0, 0.0)})}
BIN_X = W.BIN[1] if args.bin_x is None else args.bin_x
BIN_Y = W.BIN[2] if args.bin_y is None else args.bin_y
START_XY = (W.START[0] if args.start_x is None else args.start_x, W.START[1] if args.start_y is None else args.start_y)
if abs(args.shift - W.SHIFT) > 1e-6 or abs(BIN_X - W.BIN[1]) > 1e-6 or abs(BIN_Y - W.BIN[2]) > 1e-6:
    print(f"[DEMO] WARNING: bench/bin ({args.shift},{BIN_X},{BIN_Y}) differ from the Nav2 map ({W.SHIFT},{W.BIN[1]},{W.BIN[2]})", flush=True)
BIN_REL = (BIN_X, BIN_Y, BIN_H / 2)
_rz = cfg.scene.robot.init_state.pos[2]
cfg.scene.robot.init_state.pos = (START_XY[0], START_XY[1], _rz)
print(f"[DEMO] robot start ({START_XY[0]:+.2f},{START_XY[1]:+.2f}) z={_rz:.2f}", flush=True)
for _n, _cx, _cy, _sx, _sy, _h, _spawn in W.OBSTACLES:          # walls / baffles, same boxes as the Nav2 map
    if _spawn:
        setattr(cfg.scene, _n, AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + _n.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_cx, _cy, _h / 2)),
            spawn=sim_utils.CuboidCfg(size=(_sx, _sy, _h), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)))))
if not args.no_surprise:
    for _n, _cx, _cy, _sx, _sy, _h, _spawn in W.SURPRISE:      # in the sim, NOT in the Nav2 map -> only the scan sees it
        setattr(cfg.scene, _n, AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/" + _n.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_cx, _cy, _h / 2)),
            spawn=sim_utils.CuboidCfg(size=(_sx, _sy, _h), collision_props=sim_utils.CollisionPropertiesCfg(),
                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05)))))
    print(f"[SCAN] surprise obstacles (in the sim, NOT in the Nav2 map): {[(n, cx, cy) for n, cx, cy, *_ in W.SURPRISE]}", flush=True)
USE_MOVER = bool(getattr(W, "MOVER", None)) and not args.no_mover
if USE_MOVER:                                 # kinematic box: moved by code each step, cannot be pushed, seen by the PhysX lidar
    from isaaclab.assets import RigidObjectCfg
    _m = W.MOVER
    cfg.scene.mover = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Mover",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(_m["start"][0], _m["start"][1], _m["size"][2] / 2)),
        spawn=sim_utils.CuboidCfg(size=tuple(_m["size"]),
                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=20.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg(),
                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))))
    print(f"[MOVER] box {_m['size']} from {_m['start']} to {_m['target']} at {_m['speed']} m/s, triggered in {_m['phase']} within {_m['trigger']} m (NOT in the Nav2 map)", flush=True)
cfg.scene.bin = AssetBaseCfg(
    prim_path="{ENV_REGEX_NS}/Bin",
    init_state=AssetBaseCfg.InitialStateCfg(pos=BIN_REL),
    spawn=UsdFileCfg(usd_path="@SPOT_ASSETS@/props/small_KLT_hollow.usd", scale=(args.bin_scale, args.bin_scale, 1.0)),
)
for _bn, _bx, _by, *_br in getattr(W, "EXTRA_BINS", []):      # PLACE: extra bins, same KLT as the main bin
    setattr(cfg.scene, _bn, AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + _bn.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_bx, _by, BIN_H / 2)),
        spawn=UsdFileCfg(usd_path="@SPOT_ASSETS@/props/small_KLT_hollow.usd", scale=(args.bin_scale, args.bin_scale, 1.0))))
# ---- PERC: hand camera on the wrist (real Spot-like FOV 56x44 deg), debug markers off ----
cfg.scene.ee_frame.debug_vis = False
for _k, _c in vars(cfg.commands).items():
    if hasattr(_c, "debug_vis"):
        _c.debug_vis = False
cfg.scene.hand_cam = CameraCfg(
    prim_path=f"{cfg.scene.robot.prim_path}/arm0_link_wr1/hand_cam",
    update_period=0.0, height=480, width=640,
    data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
    colorize_instance_id_segmentation=False,
    spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=10.63, clipping_range=(0.05, 5.0)),
    offset=CameraCfg.OffsetCfg(pos=tuple(args.cam_pos), convention="world"),
)
cfg.scene.hand_cam.update_latest_camera_pose = True
if args.body_cam:                                  # BODY CAM: front camera on the body, looking forward and down
    _bp = math.radians(args.body_cam_pitch)
    from isaaclab.utils.math import quat_from_euler_xyz as _qe          # quaternion in Isaac Lab's own order (3.0: x,y,z,w)
    _bq = tuple(float(v) for v in _qe(torch.tensor([0.0]), torch.tensor([_bp]), torch.tensor([0.0]))[0])
    cfg.scene.body_cam = CameraCfg(
        prim_path=f"{cfg.scene.robot.prim_path}/{args.body_cam_link}/body_cam",
        update_period=0.0, height=480, width=640,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0 * math.tan(math.radians(args.body_cam_hfov) / 2),
                                         clipping_range=(0.05, 6.0)),
        offset=CameraCfg.OffsetCfg(pos=tuple(args.body_cam_pos), rot=_bq, convention="world"),
    )
    cfg.scene.body_cam.update_latest_camera_pose = True
    print(f"[BODYCAM] front body camera on {args.body_cam_link} at {args.body_cam_pos}, {args.body_cam_pitch:.0f} deg down, {args.body_cam_hfov:.0f} deg wide", flush=True)
if args.neighbours:
    from isaaclab.assets import RigidObjectCfg
    for _name, _col, _dy in (("CubeRed", (0.8, 0.1, 0.1), 0.30), ("CubeBlue", (0.1, 0.2, 0.8), -0.30)):
        setattr(cfg.scene, _name.lower(), RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/" + _name,
            init_state=RigidObjectCfg.InitialStateCfg(pos=(BENCH_X + args.shift, _dy, CUBE_Z)),
            spawn=sim_utils.CuboidCfg(
                size=(0.048, 0.048, 0.048),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_col))))
    print("[PERC] TEST scene: red cube +0.30 m, blue cube -0.30 m beside the task cube", flush=True)
env = gym.make(task, cfg=cfg)
u = env.unwrapped
dev, dt = u.device, u.step_dt
robot, obj, ee = u.scene["robot"], u.scene["object"], u.scene["ee_frame"]
arm_term, legs_term = u.action_manager.get_term("arm_action"), u.action_manager.get_term("legs")
arm_names = list(arm_term._joint_names)
arm_ids = robot.find_joints(arm_names, preserve_order=True)[0]
f_id = robot.find_joints("arm0_f1x")[0][0]
ready = arm_term._offset[0].clone()
stow = torch.tensor([STOW[n] for n in arm_names], device=dev)
place = torch.tensor([PLACE[n] for n in arm_names], device=dev)
PLACE0 = place.clone()
policy = torch.jit.load(args.arm, map_location=dev).eval()
# ---------------- PERC: perception helpers ----------------
cam = u.scene["hand_cam"]
BODY_CAM = u.scene["body_cam"] if args.body_cam else None     # BODY CAM handle
SEL = {"colour": None, "prim": "/env_0/Object/"}            # FLOW: which cube the hand camera / arm work on
_om = u.observation_manager
_names = _om.active_terms["policy"]; _dims = [d[0] for d in _om.group_obs_term_dim["policy"]]
I0 = sum(_dims[:_names.index("object_position")])
WR1 = robot.find_bodies("arm0_link_wr1")[0][0]
GOFS = torch.tensor(args.gofs, device=dev)
CUBE_HALF = 0.024
try:
    _w = robot.find_joints("arm0_wr0")[0][0]
    _pl = getattr(robot.data, "joint_pos_limits", None)
    _pl = T(_pl if _pl is not None else robot.data.soft_joint_pos_limits)[0]
    _el = getattr(robot.data, "joint_effort_limits", None)
    _el = f"{float(T(_el)[0, _w]):.1f}" if _el is not None else "n/a"
    print(f"[PERC] wr0 position limits {float(_pl[_w, 0]):+.2f} .. {float(_pl[_w, 1]):+.2f} rad | effort limit {_el}", flush=True)
except Exception as e:
    print(f"[PERC] could not read wr0 limits: {e}", flush=True)
print(f"[PERC] object_position at obs index {I0}..{I0 + 2} | wrist body index {WR1} | grasp offset {args.gofs}", flush=True)
_vv, _uu = torch.meshgrid(torch.arange(480, device=dev, dtype=torch.float32),
                          torch.arange(640, device=dev, dtype=torch.float32), indexing="ij")
P = {"frames": [], "est_w": None, "att": False, "stall": 0, "wz0": 0.0, "z0": 0.0, "gofs": GOFS, "nohand": 0, "tries": 0, "why": "", "vis": {"hidden": 0, "edge": 0, "notop": 0}, "look_i": 0, "reparks": 0, "repark_xy": None}

def _id_map(info):
    d = info[0] if isinstance(info, (list, tuple)) else info
    d = d.get("instance_id_segmentation_fast", d) if isinstance(d, dict) else d
    return d.get("idToLabels", d) if isinstance(d, dict) else {}

def _cube_mask():
    out = cam.data.output
    dep = T(out["distance_to_image_plane"])[0, ..., 0].float()
    seg = T(out["instance_id_segmentation_fast"])[0, ..., 0].long()
    ids = torch.tensor([int(k) for k, v in _id_map(cam.data.info).items() if SEL["prim"] in str(v)],   # FLOW: selected cube
                       device=dev, dtype=torch.long)
    return dep, torch.isin(seg, ids) & torch.isfinite(dep) & (dep > 0)

def cube_from_frame():
    """One camera frame -> cube centre in world, or None if hidden / cut by the image edge / no top face."""
    dep, mask = _cube_mask()
    if int(mask.sum()) < 30:
        P["vis"]["hidden"] += 1
        return None
    if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        P["vis"]["edge"] += 1
        return None
    K = T(cam.data.intrinsic_matrices)[0]; pos = T(cam.data.pos_w)[0]; q = T(cam.data.quat_w_ros)[0]
    d = dep[mask]
    pc = torch.stack([(_uu[mask] - K[0, 2]) * d / K[0, 0], (_vv[mask] - K[1, 2]) * d / K[1, 1], d], dim=-1)
    pw = quat_apply(q.unsqueeze(0).expand(pc.shape[0], 4), pc) + pos
    top = pw[:, 2] > torch.quantile(pw[:, 2], 0.95) - 0.01
    if int(top.sum()) < 50:
        P["vis"]["notop"] += 1
        return None
    _xy = pw[top, :2] - pw[top, :2].mean(dim=0)
    _w = ((_xy ** 2).sum(dim=1)) ** 2                       # r^4: corners dominate
    _a = torch.atan2(_xy[:, 1], _xy[:, 0])
    _corner = math.atan2(float((_w * torch.sin(4 * _a)).sum()), float((_w * torch.cos(4 * _a)).sum()))
    _yaw = (_corner - math.pi) / 4.0                          # square to the axes -> 0
    _yaw = (_yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4
    P.setdefault("yaw_frames", []).append(_yaw)
    _ys, _xs = torch.nonzero(mask, as_tuple=True)
    _y0, _y1, _x0, _x1 = int(_ys.min()), int(_ys.max()) + 1, int(_xs.min()), int(_xs.max()) + 1
    if _y1 - _y0 > 4 and _x1 - _x0 > 4:
        _dc, _mc = dep[_y0:_y1, _x0:_x1], mask[_y0:_y1, _x0:_x1]
        _uc, _vc = _uu[_y0:_y1, _x0:_x1], _vv[_y0:_y1, _x0:_x1]
        _pc = torch.stack([(_uc - K[0, 2]) * _dc / K[0, 0], (_vc - K[1, 2]) * _dc / K[1, 1], _dc], dim=-1)
        _du = _pc[1:-1, 2:] - _pc[1:-1, :-2]
        _dv = _pc[2:, 1:-1] - _pc[:-2, 1:-1]
        _inner = _mc[1:-1, 1:-1] & _mc[1:-1, 2:] & _mc[1:-1, :-2] & _mc[2:, 1:-1] & _mc[:-2, 1:-1]
        _n = torch.cross(_du, _dv, dim=-1)[_inner]
        if _n.shape[0] > 30:
            _n = _n / _n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            _up = quat_apply(q.unsqueeze(0).expand(_n.shape[0], 4), _n)[:, 2].abs()
            _sel = _up > 0.7                                   # faces pointing roughly up
            if int(_sel.sum()) > 30:
                P.setdefault("tilt_frames", []).append(math.degrees(math.acos(min(1.0, float(_up[_sel].median())))))
    return torch.stack([pw[top, 0].mean(), pw[top, 1].mean(), pw[top, 2].median() - CUBE_HALF])

def wrist_pose():
    bd = robot.data
    bp = T(bd.body_link_pos_w if hasattr(bd, "body_link_pos_w") else bd.body_pos_w)[0, WR1]
    bq = T(bd.body_link_quat_w if hasattr(bd, "body_link_quat_w") else bd.body_quat_w)[0, WR1]
    return bp, bq

_sh0 = arm_names.index("arm0_sh0")
def _pan(dq):
    q = ready.clone(); q[_sh0] += dq
    return q
LOOK_POSES = [ready, _pan(0.20), _pan(-0.20)]

def save_cam_image(tag):
    try:
        from PIL import Image
        import numpy as _np, os as _os
        d = _os.path.expanduser("@SPOT_DEMO_ROOT@/logs/perception/demo_frames"); _os.makedirs(d, exist_ok=True)
        rgb = T(cam.data.output["rgb"])[0].cpu().numpy()[..., :3]
        p = f"{d}/{tag}_{int(time.time())}.png"
        Image.fromarray(rgb if rgb.dtype == _np.uint8 else (_np.clip(rgb, 0, 1) * 255).astype(_np.uint8)).save(p)
        print(f"[PERC] saved camera image {p}", flush=True)
    except Exception as e:
        print(f"[PERC] could not save camera image: {e}", flush=True)

def repark_cmd(target_xy, target_yaw):
    """Small base correction with a minimum speed (walking policy ignores < ~0.1 m/s)."""
    p, yaw = base_pose()
    d = target_xy - p
    dist = float(torch.norm(d))
    vx = vy = 0.0
    if dist > 0.02:
        sp = min(0.30, max(args.repark_vmin, 2.0 * dist))
        vx, vy = to_body(float(d[0]) / dist * sp, float(d[1]) / dist * sp, yaw)
    dyaw = wrap(target_yaw - yaw)
    wz = max(-0.4, min(0.4, 1.5 * dyaw)) if abs(dyaw) > 0.025 else 0.0
    return [vx, vy, wz], dist

_SH0B = robot.find_bodies("arm0_link_sh0")[0][0]
def shoulder_xy():
    bd = robot.data
    return T(bd.body_link_pos_w if hasattr(bd, "body_link_pos_w") else bd.body_pos_w)[0, _SH0B, :2]

def carried_cube_w():
    """Where the held cube is, from the wrist pose and the offset measured at this grasp (no simulator truth)."""
    bp, bq = wrist_pose()
    return bp + quat_apply(bq.unsqueeze(0), P["gofs"].unsqueeze(0))[0]

def hand_check():
    """Hand camera: is the (selected) cube still between the fingers? Uses _cube_mask(), so it follows the selection."""
    dep, mask = _cube_mask()
    px = int(mask.sum())
    dm = float(dep[mask].median()) if px else float("nan")
    return (px >= args.inhand_px // 2 and dm < args.inhand_d), px, dm

def colour_name(col):
    """Mean RGB (0-1) -> colour word by hue; low saturation -> 'grey'."""
    r, g, b = (float(x) for x in col)
    mx, mn = max(r, g, b), min(r, g, b)
    if mx < 1e-6 or (mx - mn) / mx < 0.15:
        return "grey"
    if mx == r:
        h = (60.0 * (g - b) / (mx - mn)) % 360.0
    elif mx == g:
        h = 60.0 * (b - r) / (mx - mn) + 120.0
    else:
        h = 60.0 * (r - g) / (mx - mn) + 240.0
    for nm, lo, hi in (("red", 0, 20), ("orange", 20, 45), ("yellow", 45, 70), ("green", 70, 160),
                       ("cyan", 160, 200), ("blue", 200, 260), ("purple", 260, 345)):
        if lo <= h < hi:
            return nm
    return "red"

_COLOURS = {"red": (0.8, 0.1, 0.1), "blue": (0.1, 0.2, 0.8), "green": (0.1, 0.7, 0.1), "yellow": (0.8, 0.8, 0.1)}
def detect_cubes():
    """Hand camera at the current pose -> {colour: (x, y)} in the map frame (sim world minus env origin).
    Object masks from sim instance segmentation (a real robot needs a detector here); colour from the RGB pixels
    (chromaticity, so lighting matters less); position from depth: top face -> centre. Objects taller than 12 cm skipped."""
    out = cam.data.output
    dep = T(out["distance_to_image_plane"])[0, ..., 0].float()
    seg = T(out["instance_id_segmentation_fast"])[0, ..., 0].long()
    rgb = T(out["rgb"])[0, ..., :3].float()
    if float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    K = T(cam.data.intrinsic_matrices)[0]; pos = T(cam.data.pos_w)[0]; q = T(cam.data.quat_w_ros)[0]
    org = T(u.scene.env_origins)[0]
    found = {}
    for k, v in _id_map(cam.data.info).items():
        lab = str(v)
        if "/env_0/" not in lab or "/Robot/" in lab or "Table" in lab or "Bin" in lab:
            continue
        mask = (seg == int(k)) & torch.isfinite(dep) & (dep > 0)
        n = int(mask.sum())
        if n < 30:
            continue
        d = dep[mask]
        pc = torch.stack([(_uu[mask] - K[0, 2]) * d / K[0, 0], (_vv[mask] - K[1, 2]) * d / K[1, 1], d], dim=-1)
        pw = quat_apply(q.unsqueeze(0).expand(pc.shape[0], 4), pc) + pos
        if float(pw[:, 2].max() - pw[:, 2].min()) > 0.12:
            continue
        top = pw[:, 2] > torch.quantile(pw[:, 2], 0.95) - 0.01
        c = torch.stack([pw[top, 0].mean(), pw[top, 1].mean(), pw[top, 2].median() - CUBE_HALF])
        col = rgb[mask].mean(0)
        name = colour_name(col)
        key = name if name not in found else f"{name}_{len(found)}"
        edge = bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
        found[key] = (round(float(c[0] - org[0]), 3), round(float(c[1] - org[1]), 3))
        print(f"[DETECT] {key}: map ({found[key][0]:+.3f}, {found[key][1]:+.3f}) z {float(c[2]):.3f} | {n} px"
              f"{' (cut by image edge)' if edge else ''} | rgb ({float(col[0]):.2f},{float(col[1]):.2f},{float(col[2]):.2f}) | {lab}", flush=True)
    return found

def give_up(reason, tag="FAILED", keep_hold=False):
    """Every give-up goes through here. tag: FAILED / SKIPPED. keep_hold: cube still in the gripper (do not open)."""
    print(f"[RESULT] {tag}: {reason}", flush=True)
    state["giveup_hold"] = keep_hold
    pp, py = base_pose()
    enter("GIVEUP", hold_xy=pp, hold_yaw=py)            # base held where it is

def after_giveup():
    """Default: end the run (single-cube demo). The nav multi_flow patch replaces this."""
    enter("DONE")

def retry_or_stop(reason):
    P["tries"] += 1
    P["look_i"] = 0
    P["reparks"] = 0
    if P["tries"] > args.retries:
        give_up(f"{reason} - no retries left")
    else:
        print(f"[PERC] {reason} - retry {P['tries']}/{args.retries}", flush=True)
        enter("RETRY")

def perc_look_step(t):
    if t <= 1:                       # enter() sets t=0, the main loop adds 1 in the same step -> first LOOK step has t=1
        P.update(frames=[], est_w=None, att=False, stall=0, nohand=0, why="", vis={"hidden": 0, "edge": 0, "notop": 0})
        state["via_done"] = False
        state["place_fixes"] = 0
        P["yaw_frames"] = []
        P["veto"] = 0
        place[:] = PLACE0                                   # undo last cycle's sideways fix
        P["tilt_frames"] = []
        if args.detect_test and P.get("look_i", 0) == 0:
            print(f"[DETECT] result: {detect_cubes()}", flush=True)
    if t >= int(0.1 / dt):
        r = cube_from_frame()
        if r is not None:
            P["frames"].append(r)
    if t >= int(args.look_s / dt) and (len(P["frames"]) >= args.look_min or (t * dt >= 1.0 and len(P["frames"]) >= args.look_min_late)):
        est = torch.stack(P["frames"]).median(dim=0).values
        _yf = sorted(P.get("yaw_frames", []))
        if _yf:
            _ym = math.degrees(_yf[len(_yf) // 2])
            try:
                _, _, _ty = euler_xyz_from_quat(T(obj.data.root_quat_w))
                _ty = float(_ty[0]); _tf = math.degrees((_ty + math.pi / 4) % (math.pi / 2) - math.pi / 4)
                _tt = f" | truth (log only) {_tf:+.1f} deg"
            except Exception:
                _tt = ""
            print(f"[PERC] cube yaw (perception) {_ym:+.1f} deg off square{_tt}", flush=True)
            if abs(_ym) > args.max_yaw_deg:
                give_up(f"cube rotated {_ym:+.0f} deg - can't pick (policy trained on square cubes)", tag="SKIPPED")
                return
        _tl = sorted(P.get("tilt_frames", []))
        if _tl:
            _tm = _tl[len(_tl) // 2]
            try:
                _qo = T(obj.data.root_quat_w)[0:1]
                _zc = [abs(float(quat_apply(_qo, torch.tensor([e], device=dev, dtype=torch.float32))[0, 2]))
                       for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
                _tt2 = f" | truth (log only, nearest face) {math.degrees(math.acos(min(1.0, max(_zc)))):.1f} deg"
            except Exception:
                _tt2 = ""
            print(f"[PERC] cube tilt (perception) {_tm:.1f} deg{_tt2}", flush=True)
            if _tm > args.max_tilt_deg:
                give_up(f"cube tipped {_tm:.0f} deg - can't pick (top face not level)", tag="SKIPPED")
                return
        P["est_w"] = est
        e = (est - T(obj.data.root_pos_w)[0, :3]) * 1000
        print(f"[PERC] look: {len(P['frames'])} good frames | estimate vs truth (log only) "
              f"dx {float(e[0]):+.1f} dy {float(e[1]):+.1f} dz {float(e[2]):+.1f} mm", flush=True)
        state["held"] = 0
        cb = subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), est.unsqueeze(0))[0][0]
        off_x, off_y = float(cb[0]) - args.pick_x, float(cb[1])
        print(f"[PERC] cube in base frame x {float(cb[0]):.3f} y {float(cb[1]):+.3f} z {float(cb[2]):.3f} m "
              f"(wanted x {args.pick_x:.3f} y 0)", flush=True)
        if (abs(off_x) > args.pick_tol_x or abs(off_y) > args.pick_tol_y) and P["reparks"] < 2:
            p, yaw = base_pose()
            dxw = math.cos(yaw) * off_x - math.sin(yaw) * off_y
            dyw = math.sin(yaw) * off_x + math.cos(yaw) * off_y
            P["repark_xy"] = p + torch.tensor([dxw, dyw], device=dev)
            P["reparks"] += 1
            print(f"[PERC] re-park {P['reparks']}/2: move base {off_x:+.3f} forward, {off_y:+.3f} left (m)", flush=True)
            enter("REPARK", hold_xy=P["repark_xy"], hold_yaw=state["hold_yaw"])
        else:
            enter("PICK" if P["look_i"] == 0 else "UNLOOK")
    elif t * dt > 2.0:
        print(f"[PERC] look pose {P['look_i']}: cube not seen well enough ({len(P['frames'])} good frames in 2 s, rejects {P['vis']})", flush=True)
        save_cam_image(f"lookfail_pose{P['look_i']}")
        P["look_i"] += 1
        if P["look_i"] < len(LOOK_POSES):
            enter("LOOKMOVE")
        else:
            give_up("no look pose sees the cube")

def perc_pick_step(obs):
    q = float(T(robot.data.joint_pos)[0, f_id]); qd = float(T(robot.data.joint_vel)[0, f_id])
    bp, bq = wrist_pose()
    fail = False
    if not P["att"]:
        P["stall"] = P["stall"] + 1 if (-0.9 < q < -0.2 and abs(qd) < 0.5) else 0
        if P["stall"] >= args.stall_steps:
            g = quat_apply(quat_conjugate(bq.unsqueeze(0)), (P["est_w"] - bp).unsqueeze(0))[0]
            if args.const_gofs or float((g - GOFS).norm()) > 0.05:
                src = "fixed" if args.const_gofs else f"fixed (per-grasp {[round(v, 3) for v in g.tolist()]} off by >5 cm)"
                g = GOFS
            else:
                src = "this grasp"
            P["gofs"], P["att"], P["nohand"], P["z0"] = g, True, 0, float(P["est_w"][2])
            print(f"[PERC] grasp detected (finger {q:.2f} rad) | offset in wrist frame {[round(v, 3) for v in g.tolist()]} from {src}", flush=True)
    elif q < -1.2 or q > -0.1:
        P["att"], P["stall"] = False, 0
        fail, P["why"] = True, f"grasp ended (finger {q:.2f} rad)"
    if P["att"]:
        P["est_w"] = bp + quat_apply(bq.unsqueeze(0), P["gofs"].unsqueeze(0))[0]
        P["att_t"] = P.get("att_t", 0) + 1
        if P["att_t"] >= int(args.grasp_check_s / dt):
            _ok, _px, _dm = hand_check()
            P["nogrip"] = 0 if _ok else P.get("nogrip", 0) + 1
            if P["nogrip"] >= HOLD and not fail:
                fail, P["why"] = True, (f"false grasp: finger closed but hand camera sees no cube between the fingers "
                                        f"({_px} px, depth {_dm:.2f} m)")
    else:
        P["att_t"], P["nogrip"] = 0, 0
    seen = subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), P["est_w"].unsqueeze(0))[0]
    o = obs["policy"].clone()
    o[:, I0:I0 + 3] = seen
    a = policy(o)[:, :7]
    if args.gate and not P["att"] and float(a[0, 6]) < 0:
        _loc = quat_apply(quat_conjugate(bq.unsqueeze(0)), (P["est_w"] - bp).unsqueeze(0))[0]
        if abs(float(_loc[1]) - args.gate_y0) > args.gate_y or abs(float(_loc[2]) - args.gate_z0) > args.gate_z:
            a[0, 6] = 1.0                                       # keep the gripper open: cube off-centre
            P["veto"] = P.get("veto", 0) + 1
            if P["veto"] == 1:
                print(f"[PERC] close vetoed: cube y {float(_loc[1]):+.3f} z {float(_loc[2]):+.3f} m in the gripper "
                      f"(allowed y {args.gate_y0:+.3f}+-{args.gate_y:.3f}, z {args.gate_z0:+.3f}+-{args.gate_z:.3f})", flush=True)
            if P["veto"] * dt > args.gate_max_s and not fail:
                fail, P["why"] = True, f"policy kept closing off-centre ({float(_loc[1]):+.3f} m) - cube left untouched"
    ok, px, dmed = False, 0, float("nan")
    rise = float(P["est_w"][2]) - P["z0"] if P["att"] else 0.0
    if P["att"] and rise > args.lift_check:
        dep, mask = _cube_mask()
        px = int(mask.sum())
        dmed = float(dep[mask].median()) if px else float("nan")
        ok = px >= args.inhand_px and dmed < args.inhand_d
        P["nohand"] = 0 if ok else P["nohand"] + 1
        if P["nohand"] >= HOLD:
            fail, P["why"] = True, f"lifted but camera does not see the cube in hand (px {px}, depth {dmed:.2f})"
    err = float((P["est_w"] - T(obj.data.root_pos_w)[0, :3]).norm()) * 100
    P["last"] = (px, dmed, rise)
    state["info"] = f"att={int(P['att'])} rise={rise:.2f} cam_px={px} cam_d={dmed:.2f} est_err={err:.1f}cm(log)"
    return a, ok, fail
train_kp = train_kd = None
try:
    train_kp = T(robot.data.joint_stiffness)[0, arm_ids].clone()
    train_kd = T(robot.data.joint_damping)[0, arm_ids].clone()
except Exception as e:
    print(f"[DEMO] could not read training gains ({e}); using 400/40", flush=True)
    train_kp = torch.full((len(arm_ids),), 400.0, device=dev)
    train_kd = torch.full((len(arm_ids),), 40.0, device=dev)
big_ids = robot.find_joints(["arm0_sh0", "arm0_sh1", "arm0_el0"], preserve_order=True)[0]
try:
    _GRIP_KP0 = float(T(robot.data.joint_stiffness)[0, f_id])
    _gel = getattr(robot.data, "joint_effort_limits", None)
    print(f"[PERC] finger arm0_f1x: training stiffness {_GRIP_KP0:.0f}, effort limit "
          f"{float(T(_gel)[0, f_id]) if _gel is not None else float('nan'):.1f} | carry stiffness {args.grip_kp:.0f}", flush=True)
except Exception as e:
    _GRIP_KP0 = 400.0
    print(f"[PERC] could not read finger gains ({e}); assuming 400", flush=True)
_gain_mode = [None]
_EFF0 = None
def set_wrist_effort(val):
    """Wrist torque limit (sim). val = float, or None to restore the original limits."""
    global _EFF0
    ids = robot.find_joints(["arm0_wr0", "arm0_wr1"], preserve_order=True)[0]
    fn = getattr(robot, "write_joint_effort_limit_to_sim_index", None) or getattr(robot, "write_joint_effort_limit_to_sim", None)
    if fn is None:
        print("[DEMO] no effort-limit writer in this Isaac Lab version - --wrist_effort ignored", flush=True)
        return
    if _EFF0 is None:
        _EFF0 = T(robot.data.joint_effort_limits)[0, ids].clone()
    lim = _EFF0.unsqueeze(0) if val is None else val
    try:
        fn(limits=lim, joint_ids=ids)
    except TypeError:
        fn(lim, joint_ids=ids)

def apply_gains(phase):
    """Stiff + fast for scripted moves; the policy's own training gains while it is in control."""
    if args.fast_kp is None:
        return
    mode = "train" if phase in ("SETTLE", "LOOK", "PICK") else "fast"
    if mode == _gain_mode[0]:
        return
    if mode == "fast":
        robot.write_joint_stiffness_to_sim_index(stiffness=train_kp.unsqueeze(0), joint_ids=arm_ids)   # wrist etc. = training gains
        robot.write_joint_damping_to_sim_index(damping=train_kd.unsqueeze(0), joint_ids=arm_ids)
        robot.write_joint_stiffness_to_sim_index(stiffness=args.fast_kp, joint_ids=big_ids)             # shoulder + elbow = fast
        robot.write_joint_damping_to_sim_index(damping=args.fast_kd, joint_ids=big_ids)
        if args.wrist_kp is not None:
            _wids = robot.find_joints(["arm0_wr0", "arm0_wr1"], preserve_order=True)[0]
            robot.write_joint_stiffness_to_sim_index(stiffness=args.wrist_kp, joint_ids=_wids)
            robot.write_joint_damping_to_sim_index(damping=args.wrist_kd, joint_ids=_wids)
        if args.wrist_effort is not None:
            set_wrist_effort(args.wrist_effort)
        if args.grip_kp is not None:
            robot.write_joint_stiffness_to_sim_index(stiffness=args.grip_kp, joint_ids=[f_id])   # firmer hold
        msg = f"shoulder+elbow kp={args.fast_kp:.0f} kd={args.fast_kd:.0f}, wrist at training gains"
    else:
        robot.write_joint_stiffness_to_sim_index(stiffness=train_kp.unsqueeze(0), joint_ids=arm_ids)
        robot.write_joint_damping_to_sim_index(damping=train_kd.unsqueeze(0), joint_ids=arm_ids)
        if args.wrist_effort is not None:
            set_wrist_effort(None)
        if args.grip_kp is not None:
            robot.write_joint_stiffness_to_sim_index(stiffness=_GRIP_KP0, joint_ids=[f_id])      # training value
        msg = f"policy training gains kp={float(train_kp[0]):.0f} kd={float(train_kd[0]):.0f}"
    _gain_mode[0] = mode
    print(f"[DEMO] arm gains -> {mode} ({msg})", flush=True)
origin = T(u.scene.env_origins)[0, :2]
from goto_planner import GotoPlanner, pick_headings, place_headings
gp = GotoPlanner(W.OBSTACLES, pre_dist=args.pre_dist)
_pp = gp.plan((W.BENCH[1], W.BENCH[2]) if args.multi_flow else CUBE_XY, W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")    # MULTI/FLOW
for _t in _pp["tried"]:
    print(f"[GOTO]   pick  {_t}", flush=True)
if not _pp["ok"]:
    print(f"[RESULT] NAV FAILED: {_pp['reason']} - change the scene (check_scene.py)", flush=True)
    env.close(); app.close(); sys.exit(1)
PICK_YAW = _pp["stand"][2]
bench_park = origin + torch.tensor(_pp["stand"][:2], device=dev)
bench_park = bench_park - torch.tensor([args.park_back, 0.0], device=dev)
bin_w = origin + torch.tensor(BIN_REL[:2], device=dev)
_pl = gp.plan((BIN_X, BIN_Y), args.place_dist, place_headings(), robot_xy=_pp["stand"][:2], label="place")
for _t in _pl["tried"]:
    print(f"[GOTO]   place {_t}", flush=True)
if not _pl["ok"]:
    print(f"[RESULT] NAV FAILED: {_pl['reason']} - change the scene (check_scene.py)", flush=True)
    env.close(); app.close(); sys.exit(1)
PLACE_YAW = _pl["stand"][2]
bin_park = origin + torch.tensor(_pl["stand"][:2], device=dev)
print(f"[GOTO] {_pp['reason']} | {_pl['reason']}", flush=True)

# ---- NAV: link to the ROS 2 side (relay + Nav2 + goal bridge) ----
_box_scan = Scan2D(W.OBSTACLES, n_rays=args.scan_rays)
if args.scan == "physx":
    from phys_scan import PhysxScan
    import re as _re
    _robot_path = cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
    _robot_path = _re.sub(r"env_[^/]*", "env_0", _robot_path)          # regex form env_.* -> env_0
    print(f"[SCAN] excluding Spot's own colliders under {_robot_path} (cfg prim_path {cfg.scene.robot.prim_path})", flush=True)
    _scan = PhysxScan(origin_xy=(float(origin[0]), float(origin[1])), exclude=(_robot_path, "/World/envs/env_0/Mover"),
                      n_rays=args.scan_rays, height=args.scan_h)
else:
    _scan = _box_scan
print(f"[SCAN] scan source: {args.scan} | {args.scan_rays} rays | plane z={args.scan_h} m", flush=True)
link = StateLink(W.OBSTACLES, dt, scan_fn=_scan)
odom = OdomDrift(dt, scale=args.odom_scale, yaw_bias=args.odom_yaw_bias, noise=args.odom_noise)
print(f"[ODOM] {'exact (no drift)' if args.no_drift else f'drifting: scale {args.odom_scale:+.0%}, yaw bias {math.degrees(args.odom_yaw_bias):.2f} deg/s, noise {args.odom_noise:.0%}'}"
      f" - Nav2 localisation needs nav_up.sh --amcl", flush=True)
hh = HeadingHold(k=3.0, wz_bias=args.wz_bias)
goals = GoalClient()
prepick = origin + torch.tensor(_pp["pre"][:2], device=dev)      # GOTO: checked pre-poses (room to turn)
preplace = origin + torch.tensor(_pl["pre"][:2], device=dev)
PLACES = {"bin": (bin_w.clone(), bin_park.clone(), preplace.clone(), PLACE_YAW)}     # PLACE: name -> (centre, stand, pre-pose, yaw)
for _bn, _bx, _by, *_br in getattr(W, "EXTRA_BINS", []):
    _r2 = gp.plan((_bx, _by), args.place_dist, place_headings(), robot_xy=_pp["stand"][:2], label=_bn)
    if _r2["ok"]:
        PLACES[_bn] = (origin + torch.tensor([_bx, _by], device=dev), origin + torch.tensor(_r2["stand"][:2], device=dev),
                       origin + torch.tensor(_r2["pre"][:2], device=dev), _r2["stand"][2])
        print(f"[PLACE] {_r2['reason']}: stand ({_r2['stand'][0]:+.2f},{_r2['stand'][1]:+.2f}) pre-pose ({_r2['pre'][0]:+.2f},{_r2['pre'][1]:+.2f})", flush=True)
    else:
        print(f"[PLACE] {_bn}: no stand pose ({_r2['reason']}) - not offered", flush=True)
print(f"[PLACE] place targets: {list(PLACES)}", flush=True)
def rel(v):
    return round(float(v[0] - origin[0]), 3), round(float(v[1] - origin[1]), 3)
print(f"[DEMO] walking: {'scripted (--no_nav)' if args.no_nav else 'Nav2'} | pre-pick {rel(prepick)} yaw {math.degrees(PICK_YAW):+.0f} | "
      f"pick park {rel(bench_park)} | pre-place {rel(preplace)} yaw {math.degrees(PLACE_YAW):+.0f} | bin park {rel(bin_park)}", flush=True)

BIN_TOP, BIN_HALF = BIN_H, (0.10, 0.15)
try:
    from pxr import Usd, UsdGeom
    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
        u.sim.stage.GetPrimAtPath("/World/envs/env_0/Bin")).ComputeAlignedRange()
    sz = rng.GetSize()
    BIN_TOP, BIN_HALF = float(rng.GetMax()[2]), (float(sz[0]) / 2, float(sz[1]) / 2)
    print(f"[DEMO] bin size (m): {sz[0]:.3f} x {sz[1]:.3f} x {sz[2]:.3f}, bottom z={float(rng.GetMin()[2]):.3f}, top z={BIN_TOP:.3f}", flush=True)
except Exception as e:
    print(f"[DEMO] could not measure bin ({e}); using defaults", flush=True)

start = [args.auto]
def reader():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd); tty.setcbreak(fd)
    try:
        while not start[0]:
            if select.select([sys.stdin], [], [], 0.1)[0] and sys.stdin.read(1).lower() == "g":
                start[0] = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
threading.Thread(target=reader, daemon=True).start()

def arm_action(q, grip_open):
    a = torch.zeros(1, 7, device=dev)
    a[0, :6] = (q - ready) / ARM_SCALE
    a[0, 6] = 1.0 if grip_open else -1.0
    return a

def arm_q():
    return T(robot.data.joint_pos)[0, arm_ids]

def base_pose():
    _, _, yaw = euler_xyz_from_quat(T(robot.data.root_quat_w))
    return T(robot.data.root_pos_w)[0, :2], float(yaw[0])

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def to_body(vxw, vyw, yaw):
    return math.cos(yaw) * vxw + math.sin(yaw) * vyw, -math.sin(yaw) * vxw + math.cos(yaw) * vyw

def goto_cmd(target_xy, target_yaw):
    """Walk to a pose: smooth, slows down, full stop inside the tolerance zone."""
    p, yaw = base_pose()
    d = target_xy - p
    dist, dyaw = float(torch.norm(d)), wrap(target_yaw - yaw)
    if dist < POS_TOL and abs(dyaw) < YAW_TOL:
        return [0.0, 0.0, 0.0], True, dist, dyaw
    vxw = vyw = 0.0
    if dist > POS_TOL * 0.5:
        sp = min(0.4, max(0.15, 1.2 * dist))
        vxw, vyw = float(d[0]) / dist * sp, float(d[1]) / dist * sp
    vx, vy = to_body(vxw, vyw, yaw)
    wz = max(-0.6, min(0.6, 1.5 * dyaw)) if abs(dyaw) > YAW_TOL * 0.5 else 0.0
    return [vx, vy, wz], False, dist, dyaw

def _wz_bias():
    """Yaw bias the nav build adds to scripted leg commands (--wz_bias, off with --no_script_bias); 0 standalone."""
    return 0.0 if getattr(args, "no_script_bias", False) else float(getattr(args, "wz_bias", 0.0) or 0.0)

def hold_cmd(target_xy, target_yaw, xy=True):
    """Gentle correction to stay on a pose while standing (stops slow drift of heading/position)."""
    p, yaw = base_pose()
    d = target_xy - p
    dyaw = wrap(target_yaw - yaw)
    vx = vy = 0.0
    if xy and float(torch.norm(d)) > 0.03:
        vx, vy = to_body(max(-0.2, min(0.2, float(d[0]))), max(-0.2, min(0.2, float(d[1]))), yaw)
    wz = max(-0.4, min(0.4, 1.5 * dyaw)) if abs(dyaw) > 0.025 else 0.0
    if wz != 0.0 and state.get("phase") != "PICK":       # policy ignores slow turns; never kick during the grasp
        wz = math.copysign(max(abs(wz), args.hold_wz_min), wz) - _wz_bias()   # net turn >= min after nav bias
    return [vx, vy, wz]

HOLD = int(0.5 / dt)
MOVE_STEPS = int(args.move_time / dt)
PLACE_STEPS = int(max(2.0 * args.move_time, 2.0) / dt)
VIA = 0.5 * (stow + place)                 # midpoint pose for PLACE recovery

def place_recover(reason="arm stuck"):
    state["place_tries"] = state.get("place_tries", 0) + 1
    if state["place_tries"] > 2:
        give_up(f"PLACE {reason} - no recoveries left, cube still held", keep_hold=True)
        return
    print(f"[PLACE] {reason} - recovery {state['place_tries']}/2: midpoint pose, then PLACE slower", flush=True)
    enter("PLACE_VIA")
state = {"phase": "", "t": 0, "q_from": None, "good": 0, "held": 0, "wp": [], "wp_i": 0, "info": "",
         "hold_xy": None, "hold_yaw": 0.0}
def enter(p, hold_xy=None, hold_yaw=None):
    state.update(phase=p, t=0, good=0, q_from=arm_q().clone(), best=1e9, best_t=0, goal_sent=False)
    if hold_xy is not None:
        state["hold_xy"], state["hold_yaw"] = hold_xy.clone(), hold_yaw
    print(f"[DEMO] {p}  (sim {globals().get('step', 0) * dt:.1f}s | wall {time.time() - T_START:.1f}s)", flush=True)
    if p == "LOOK" and args.multi_flow:
        try:
            _w, _x, _y, _z = [float(v) for v in obj.data.root_quat_w[0]]
            _yaw = math.degrees(math.atan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y * _y + _z * _z)))
            _tl = math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (_x * _x + _y * _y)))))
            print(f"[MULTI] {SEL['colour']} cube yaw (sim truth, log only): {_yaw:+.1f} deg | off square {((_yaw + 45) % 90) - 45:+.1f} deg | tilt {_tl:.1f} deg", flush=True)
        except Exception as _e:
            print(f"[MULTI] cube yaw log failed: {_e}", flush=True)
    apply_gains(p)

def arm_diag(target):
    d = (arm_q() - target).tolist()
    txt = " ".join(f"{n[5:]}:{e:+.2f}" for n, e in zip(arm_names, d))
    try:
        tau = T(robot.data.applied_torque)[0, arm_ids]
        lim = getattr(robot.data, "joint_effort_limits", None)
        if lim is not None:
            lim = T(lim)[0, arm_ids]
            txt += " | torque/limit " + " ".join(f"{n[5:]}:{abs(float(a)) / max(float(b), 1e-6):.2f}" for n, a, b in zip(arm_names, tau, lim))
        else:
            txt += " | torque " + " ".join(f"{n[5:]}:{float(a):+.0f}" for n, a in zip(arm_names, tau))
    except Exception as e:
        txt += f" | torque n/a ({e})"
    return txt

def arm_move(target, grip_open, next_phase, on_stuck=None, move_steps=None, wrist_tol=None, **hold):
    """Interpolate to target, then wait. Done when within ARM_TOL, or within ARM_TOL_NEAR and no longer improving.
    Far from target and not improving for ARM_STUCK_S (or ARM_MAX_WAIT) -> on_stuck() if given, else continue."""
    S, t = state, state["t"]
    if move_steps is None and not grip_open:
        move_steps = int(MOVE_STEPS * args.carry_slow)    # holding the cube: slower, gentler
    frac = min(1.0, t / (move_steps or MOVE_STEPS))
    a = arm_action(S["q_from"] + frac * (target - S["q_from"]), grip_open)
    d = (arm_q() - target).abs()
    err = float(d.max())
    vel = float(T(robot.data.joint_vel)[0, arm_ids].abs().max())
    worst = arm_names[int(d.argmax())]
    S["info"] = f"arm_err={err:.2f} arm_vel={vel:.2f} worst={worst[5:]}"
    if frac < 1.0 or err < S["best"] - 0.02:
        S["best"], S["best_t"] = err, t
    if frac < 1.0:
        return a
    still = (t - S["best_t"]) * dt > ARM_STUCK_S
    _wi = arm_names.index("arm0_wr0")
    _nw = float(torch.cat([d[:_wi], d[_wi + 1:]]).max())
    _wrist_ok = wrist_tol is not None and still and _nw < ARM_TOL and float(d[_wi]) < wrist_tol
    if _wrist_ok and err >= ARM_TOL_NEAR:
        print(f"[PLACE] wrist {float(d[_wi]):.2f} rad short, other joints on target - close enough, continuing", flush=True)
    if err < ARM_TOL or (still and err < ARM_TOL_NEAR) or _wrist_ok:
        if err >= ARM_TOL:
            print(f"[DEMO] arm settled {err:.2f} rad from target (worst {worst}) - close enough, continuing", flush=True)
        enter(next_phase, **hold)
    elif still or t * dt > ARM_MAX_WAIT:
        print(f"[DEMO] arm stuck {err:.2f} rad from target in {S['phase']} | {arm_diag(target)}", flush=True)
        if on_stuck is not None:
            on_stuck()
        else:
            enter(next_phase, **hold)
    return a

def drive(target, yaw_t, arm_q_target, grip_open, on_arrive, timeout=15.0):
    S, t = state, state["t"]
    legs, arrived, dist, dyaw = goto_cmd(target, yaw_t)
    S["info"] = f"dist={dist:.2f} dyaw={math.degrees(dyaw):+.0f}"
    if arrived:
        S["good"] += 1
    if S["good"] >= HOLD or t * dt > timeout:
        if S["good"] < HOLD:
            print(f"[DEMO] parking timeout (dist {dist:.2f} m, yaw err {math.degrees(dyaw):+.0f} deg) - continuing", flush=True)
        legs = [0.0, 0.0, 0.0]
        on_arrive()
    return arm_action(arm_q_target, grip_open), legs

def nav_phase(target_w, yaw_t, on_done, nav_cmd):
    """One Nav2 goal; follow its /cmd_vel until SUCCEEDED. On failure or timeout: log and continue with the scripted walk."""
    S = state
    if args.no_nav:
        on_done()
        return [0.0, 0.0, 0.0]
    if not S.get("goal_sent"):
        x, y = rel(target_w)
        gid = goals.go(x, y, yaw_t)
        S["goal_sent"] = True
        print(f"[NAV] Nav2 goal {gid}: ({x:+.2f},{y:+.2f}) yaw {math.degrees(yaw_t):+.0f} deg", flush=True)
    st = goals.update()
    _, yaw = base_pose()
    legs = hh(nav_cmd, yaw)                       # includes wz_bias + heading hold at zero command
    S["info"] = f"nav={st} dist_left={goals.dist:.2f} nav_cmd=({nav_cmd[0]:+.2f},{nav_cmd[2]:+.2f})"
    if st == "SUCCEEDED":
        print("[NAV] Nav2 reached the goal", flush=True)
        on_done()
    elif st in ("ABORTED", "REJECTED", "CANCELED") or S["t"] * dt > args.nav_timeout:
        print(f"[NAV] Nav2 {st} after {S['t'] * dt:.0f}s sim - continuing with the scripted walk", flush=True)
        goals.cancel()
        on_done()
    return legs

def _to_go_bin():
    enter("GO_BIN")
    state["wp"], state["wp_i"] = [(bin_park, PLACE_YAW)], 0

obs, _ = env.reset()
p0, y0 = base_pose()
# ---------------- FLOW: multi-cube user flow ----------------
CUBES = {}                                     # colour -> (scene entity, camera prim filter, (x, y) map frame)
if args.multi_flow:
    CUBES["dex"] = ("object", "/env_0/Object/", CUBE_XY)
    for _cn, _xy in DISTRACT.items():
        CUBES[_cn] = (f"cube_{_cn}", f"/env_0/Cube_{_cn}/", _xy)
REMAIN = list(CUBES)
ORDER = [c.strip().lower() for c in args.order.split(",") if c.strip()]
VIEW = (prepick.clone(), bench_park.clone(), PICK_YAW)        # stand in front of the table centre to look and ask
FLOW = {"asked": False, "reader": False, "placed": []}
def _ask_reader():
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if line.strip():
            PANEL["req"] = ("pick", line.strip().lower())
def _choose(colour, ph):
    global obj, bench_park, prepick, bin_w, bin_park, preplace, PLACE_YAW
    _parts = colour.replace(":", " ").split()                   # PLACE: "red bin2" / "red:bin2"
    colour = _parts[0] if _parts else colour
    if colour not in REMAIN:                   # CAMERA: accept the camera colour word too ("green" -> dex)
        colour = next((k for k in REMAIN if k in DET and DET[k]["colour"] == colour), colour)
    _tgt = _parts[1] if len(_parts) > 1 else FLOW.get("place_next", "bin")
    if _tgt not in PLACES:
        print(f"[MULTI] WARNING: no place called '{_tgt}' in this scene ({list(PLACES)}) - using bin", flush=True)
        _tgt = "bin"
    if ph != "ASK":
        _say(f"not asking right now ({ph})"); return
    if colour not in REMAIN:
        _say(f"no '{colour}' cube on the table - choose from {REMAIN}"); return
    ent, prim, xy = CUBES[colour]
    if (args.cam_detect or args.body_cam) and colour in DET:      # CAMERA: plan the stand pose from where the camera saw the cube
        xy = _det_xy(colour)
        print(f"[MULTI] {colour}: planning from the camera position ({xy[0]:+.3f},{xy[1]:+.3f})", flush=True)
    r = gp.plan(xy, W.STANDOFF + args.stand_extra, [PICK_YAW], label=f"pick {colour}")   # STANDOFF: --stand_extra
    if not r["ok"]:
        _say(f"cannot stand in front of {colour}: {r['tried']}"); return
    obj = u.scene[ent]
    bin_w, bin_park, preplace, PLACE_YAW = [v.clone() if torch.is_tensor(v) else v for v in PLACES[_tgt]]
    FLOW["target"] = _tgt
    SEL.update(colour=colour, prim=prim)
    bench_park = origin + torch.tensor(r["stand"][:2], device=dev)
    prepick = origin + torch.tensor(r["pre"][:2], device=dev)
    P["tries"] = 0; state["place_tries"] = 0
    P["look_i"] = 0; P["reparks"] = 0; state["giveup_hold"] = False      # FLOW: fresh perception counters per cube
    pp, _ = base_pose()
    far = float(torch.norm(bench_park - pp))
    _say(f"picking the {colour} cube -> {_tgt}" + (f" - stepping {far:.2f} m to stand in front of it" if far > 0.05 else ""))
    FLOW["park"] = 0
    enter("BACK_TABLE" if far > 0.05 else "UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)
def _after_approach():
    if args.multi_flow:
        pp, _ = base_pose()
        err = float(torch.norm(bench_park - pp))
        if err > POS_TOL + 0.01 and FLOW.get("park", 0) < 2:
            FLOW["park"] = FLOW.get("park", 0) + 1
            print(f"[MULTI] parked {err * 100:.0f} cm off the stand pose - back to the pre-pose and in again ({FLOW['park']}/2)", flush=True)
            enter("REAPPROACH"); return
        if err > POS_TOL + 0.01:
            print(f"[MULTI] WARNING: still {err * 100:.0f} cm off after 2 re-approaches - continuing, perception look poses must cope", flush=True)
        FLOW["park"] = 0
    if args.multi_flow and SEL["colour"] is None:
        DET.clear(); FLOW["scan_i"] = 0
        enter("BODYLOOK" if args.body_cam else ("SCAN" if args.cam_detect else "ASK"), hold_xy=bench_park, hold_yaw=PICK_YAW)
    else:
        enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)
def _ask_tick(t):
    if t == 1:
        if args.cam_detect or args.body_cam:
            seen = ", ".join((f"{DET[c]['colour']} ({c}) at ({_det_xy(c)[0]:+.2f},{_det_xy(c)[1]:+.2f})" if c in DET else f"{c}: NOT SEEN")
                             for c in REMAIN)
            print(f"[MULTI] cubes on the table ({'body camera' if args.body_cam else f'hand camera, {len(SCAN_POSES)} scan poses'}): {seen}", flush=True)
        seen = ", ".join(f"{c} at ({CUBES[c][2][0]:+.2f},{CUBES[c][2][1]:+.2f})" for c in REMAIN)
        print(f"[MULTI] cubes on the table (STAND-IN: from the sim until camera colour detection exists): {seen}", flush=True)
        print(f"[ASK] Which cube should I pick? {' / '.join(REMAIN)}  -> panel button, type it + Enter, or --order", flush=True)
        if len(PLACES) > 1:
            print(f"[ASK] ... and where to put it: {' / '.join(PLACES)} (type e.g. 'red bin2'; default {FLOW.get('place_next', 'bin')})", flush=True)
        PANEL["msg"] = f"Which cube? {' / '.join(REMAIN)}"
        if not FLOW["reader"] and not ORDER:
            threading.Thread(target=_ask_reader, daemon=True).start(); FLOW["reader"] = True
    if ORDER and t == int(1.0 / dt):
        c = ORDER.pop(0)
        print(f"[ASK] automatic answer (--order): {c}", flush=True)
        _choose(c, "ASK")
def _next_round():
    global obj, bench_park, prepick
    if SEL["colour"] in REMAIN:
        REMAIN.remove(SEL["colour"])
        if globals().get("inside", True):
            FLOW["placed"].append(f"{SEL['colour']}->{FLOW.get('target', 'bin')}")
        else:                                          # MISSED: the cube is not in the bin
            FLOW.setdefault("failed", []).append(f"{SEL['colour']} (missed)")
    if not REMAIN:
        print(f"[MULTI] all cubes done: placed {FLOW['placed']} | skipped/failed {FLOW.get('failed', [])}", flush=True); _say("all cubes done"); return
    SEL.update(colour=None); FLOW["park"] = 0
    prepick, bench_park = VIEW[0].clone(), VIEW[1].clone()
    _say(f"back to the table - left: {REMAIN}")
    enter("BACKOFF_BIN")
if args.multi_flow:
    def after_giveup():                        # FLOW after_giveup: replaces perception's default (end the run)
        global bench_park, prepick
        c = SEL["colour"]
        FLOW.setdefault("failed", [])
        if state.get("giveup_hold"):
            print(f"[MULTI] stopping: the {c} cube is still in the gripper after a failed PLACE | placed {FLOW['placed']} | skipped/failed {FLOW['failed'] + [c]}", flush=True)
            enter("DONE"); return
        if c in REMAIN:
            REMAIN.remove(c); FLOW["failed"].append(c)
        if not REMAIN:
            print(f"[MULTI] all cubes done: placed {FLOW['placed']} | skipped/failed {FLOW['failed']}", flush=True)
            enter("DONE"); return
        SEL.update(colour=None); FLOW["park"] = 0
        prepick, bench_park = VIEW[0].clone(), VIEW[1].clone()
        print(f"[MULTI] gave up on the {c} cube - back to the table, left: {REMAIN}", flush=True)
        enter("REAPPROACH")
DET = {}                                       # CAMERA: entity key -> {"seen": [(x, y), ...], "colour": camera colour word}
SCAN_POSES = [_pan(args.scan_pan), ready.clone(), _pan(-args.scan_pan)]
def _scan_snap():
    for name, xy in (_detect_with(BODY_CAM) if args.body_cam else detect_cubes()).items():
        best, bd = None, 0.06
        for k in REMAIN:                       # tie the detection to its sim object (mask/log only), nearest within 6 cm
            _q = T(u.scene[CUBES[k][0]].data.root_pos_w)[0]
            _d = math.hypot(float(_q[0] - origin[0]) - xy[0], float(_q[1] - origin[1]) - xy[1])
            if _d < bd:
                best, bd = k, _d
        if best is None:
            print(f"[DETECT] {name} at ({xy[0]:+.3f},{xy[1]:+.3f}) is not one of the cubes left - ignored", flush=True)
            continue
        DET.setdefault(best, {"seen": [], "colour": name.split("_")[0]})["seen"].append(xy)
        print(f"[DETECT] {name} -> {best}: camera vs truth (log only) {bd * 100:.1f} cm", flush=True)
def _detect_with(c, image_tag=None):
    """BODY CAM: run perception's detect_cubes() / save_cam_image() on another camera (they read the global cam)."""
    global cam
    _keep = cam
    cam = c
    try:
        if image_tag:
            save_cam_image(image_tag)
        return detect_cubes()
    finally:
        cam = _keep
def _det_xy(k):
    v = sorted(DET[k]["seen"]); n = len(v)
    return (sorted(a for a, _ in v)[n // 2], sorted(b for _, b in v)[n // 2])
# ---------------- PANEL: operator panel (nav side) ----------------
PANEL = {"req": None, "msg": "ready: press Start (or g), or a Go button", "pre": None, "stand": None, "yaw": 0.0, "ui": None, "label": ""}
FREE_OK = ("WAIT", "IDLE", "DONE")
def _req(*r):
    PANEL["req"] = r
def _say(m):
    PANEL["msg"] = m
    print(f"[PANEL] {m}", flush=True)
def _free_after_nav():
    if not args.no_nav and goals.status != "SUCCEEDED":
        _say(f"Nav2 {goals.status} - stopped where I am"); _free_done(arrived=False); return
    if PANEL["stand"] is not None:
        enter("FINE_FREE")
    else:
        _free_done()
def _free_done(arrived=True):
    pp, py = base_pose()
    enter("IDLE", hold_xy=pp, hold_yaw=py)
    if arrived:
        _say(f"arrived: {PANEL['label']} at ({float(pp[0] - origin[0]):+.2f},{float(pp[1] - origin[1]):+.2f}) yaw {math.degrees(py):+.0f}")
def _go(label, pre_w, stand_w, yaw):
    PANEL.update(label=label, pre=pre_w, stand=stand_w, yaw=yaw)
    _say(f"going to {label}")
    enter("FOLD_FREE")
def panel_tick(ph):
    r, PANEL["req"] = PANEL["req"], None
    if r is not None:
        kind = r[0]
        if kind == "start":
            if ph == "WAIT":
                start[0] = True; _say("full demo started")
            elif ph == "IDLE":
                enter("FOLD"); _say("full demo started")
            elif ph == "DONE":
                _say("demo already finished - restart the app to run it again")
            else:
                _say(f"busy ({ph}) - Cancel first")
        elif kind == "pick":
            _choose(r[1], ph)
        elif kind == "cancel":
            if ph in ("WAIT", "IDLE", "DONE"):
                _say("nothing to cancel")
            else:
                goals.cancel()
                pp, py = base_pose()
                enter("IDLE", hold_xy=pp, hold_yaw=py)
                _say(f"cancelled in {ph} - holding position (full demo cannot resume)")
        elif ph not in FREE_OK:
            _say(f"busy ({ph}) - Cancel first")
        elif kind == "bench":
            _go("bench (pick stand pose)", prepick, bench_park, PICK_YAW)
        elif kind == "bin":
            _go("bin (place stand pose)", preplace, bin_park, PLACE_YAW)
        elif kind == "xy":
            x, y, yd = r[1], r[2], r[3]
            d, who = gp.dist(x, y)
            if d < gp.turn_r:
                _say(f"({x:+.2f},{y:+.2f}) rejected: {d:.2f} m from {who}, need {gp.turn_r} m to turn")
            else:
                _go(f"({x:+.2f},{y:+.2f}) yaw {yd:+.0f}", origin + torch.tensor([x, y], device=dev), None, math.radians(yd))
    if PANEL["ui"] is not None and step % int(0.25 / dt) == 0:
        l_ph, l_nav, l_pose, l_msg = PANEL["ui"]
        pp, py = base_pose()
        l_ph.text = f"phase: {ph}"
        l_nav.text = f"Nav2: {goals.status}" + (f"   distance left {goals.dist:.2f} m" if goals.dist >= 0 else "")
        l_pose.text = f"Spot: ({float(pp[0] - origin[0]):+.2f}, {float(pp[1] - origin[1]):+.2f})  yaw {math.degrees(py):+.0f} deg"
        l_msg.text = PANEL["msg"]
try:
    if getattr(args, "headless", False):
        raise RuntimeError("headless run")
    import omni.ui as ui
    _w = ui.Window("Spot Nav2 panel", width=360, height=440)
    with _w.frame:
        with ui.VStack(spacing=6, height=0):
            ui.Label("Spot with arm - Nav2 operator panel", height=22)
            _l_ph = ui.Label("phase: -", height=18)
            _l_nav = ui.Label("Nav2: -", height=18)
            _l_pose = ui.Label("Spot: -", height=18)
            _l_msg = ui.Label(PANEL["msg"], height=36, word_wrap=True)
            ui.Button("Start full demo  (same as g)", height=32, clicked_fn=lambda: _req("start"))
            ui.Button("Go to bench  (pick stand pose)", height=28, clicked_fn=lambda: _req("bench"))
            ui.Button("Go to bin  (place stand pose)", height=28, clicked_fn=lambda: _req("bin"))
            with ui.HStack(height=24, spacing=4):
                ui.Label("x", width=14); _fx = ui.FloatField(width=70)
                ui.Label("y", width=14); _fy = ui.FloatField(width=70)
                ui.Label("yaw", width=30); _fyaw = ui.FloatField(width=60)
            ui.Button("Go to x, y, yaw  (map frame, deg)", height=28,
                      clicked_fn=lambda: _req("xy", _fx.model.get_value_as_float(), _fy.model.get_value_as_float(), _fyaw.model.get_value_as_float()))
            ui.Button("Cancel / stop here", height=28, clicked_fn=lambda: _req("cancel"))
            if args.multi_flow:
                ui.Label("Which cube to pick (when asked):", height=18)
                with ui.HStack(height=30, spacing=4):
                    for _c in list(CUBES):
                        ui.Button(f"Pick {_c}", clicked_fn=lambda c=_c: _req("pick", c))
                if len(PLACES) > 1:
                    ui.Label("Put it in (choose before Pick):", height=18)
                    with ui.HStack(height=30, spacing=4):
                        for _b in list(PLACES):
                            ui.Button(_b, clicked_fn=lambda b=_b: (FLOW.update(place_next=b), _say(f"next cube goes to {b}")))
    PANEL["ui"] = (_l_ph, _l_nav, _l_pose, _l_msg)
    print("[PANEL] operator panel open: window \"Spot Nav2 panel\" in the Isaac viewer", flush=True)
except Exception as _e:
    print(f"[PANEL] no panel ({_e}) - keyboard g still works", flush=True)
mover = u.scene["mover"] if USE_MOVER else None
MV = {"state": "wait", "xy": list(W.MOVER["start"]) if USE_MOVER else None, "paused": False, "write": None}
def _mover_write(x, y):
    pose = T(mover.data.root_pose_w).clone()                # keep z + orientation as the sim has them (any quat order)
    pose[0, 0], pose[0, 1] = float(origin[0]) + x, float(origin[1]) + y
    if MV["write"] is None:                                  # find the call form this Isaac Lab version accepts
        for name, call in (("index(root_pose=)", lambda p: mover.write_root_pose_to_sim_index(root_pose=p)),
                           ("index(p)", lambda p: mover.write_root_pose_to_sim_index(p)),
                           ("write_root_pose_to_sim(p)", lambda p: mover.write_root_pose_to_sim(p))):
            try:
                call(pose); MV["write"] = call
                print(f"[MOVER] pose write works via {name}", flush=True); return
            except Exception as e:
                print(f"[MOVER] pose write {name} failed: {type(e).__name__}: {e}", flush=True)
        raise RuntimeError("no working pose write for the mover")
    MV["write"](pose)
def mover_step(ph, force=False):
    if mover is None or MV["state"] == "done":
        return
    m = W.MOVER
    p, _ = base_pose()
    rx, ry = float(p[0] - origin[0]), float(p[1] - origin[1])
    if MV["state"] == "wait":
        if force or (ph == m["phase"] and math.hypot(rx - m["target"][0], ry - m["target"][1]) < m["trigger"]):
            MV["state"] = "move"
            print(f"[MOVER] sliding into the route (Spot at ({rx:+.2f},{ry:+.2f}))", flush=True)
        return
    x, y = MV["xy"]
    dx, dy = m["target"][0] - x, m["target"][1] - y
    d = math.hypot(dx, dy)
    near = math.hypot(rx - x, ry - y) < m["keep_away"]
    if near != MV["paused"]:
        MV["paused"] = near
        print(f"[MOVER] {'paused - Spot too close' if near else 'moving again'}", flush=True)
    if near:
        return
    stp = m["speed"] * dt
    if d <= stp:
        x, y = m["target"]; MV["state"] = "done"
        print(f"[MOVER] in place at ({x:+.2f},{y:+.2f}) - blocking the original route", flush=True)
    else:
        x, y = x + dx / d * stp, y + dy / d * stp
    MV["xy"] = [x, y]
    _mover_write(x, y)
if mover is not None:                         # moving box -> scan from its live pose (ray casts miss moved bodies on GPU)
    _ms = Scan2D([("mover", 0.0, 0.0, W.MOVER["size"][0], W.MOVER["size"][1], 0.0, True)], n_rays=args.scan_rays)
    _base_scan = link.scan
    def _scan_mover(x, y, yaw):
        sx, sy = W.MOVER["size"][0], W.MOVER["size"][1]
        mx, my = MV["xy"]
        _ms.boxes = np.array([[mx - sx / 2, mx + sx / 2, my - sy / 2, my + sy / 2]])
        return np.minimum(_base_scan(x, y, yaw), _ms(x, y, yaw))   # nearest hit wins (box also hides what is behind it)
    link.scan = _scan_mover
    print("[MOVER] scan = PhysX ray casts (static scene) + moving box from its live pose", flush=True)
if args.mover_test:
    if mover is None:
        print("[MOVER] test: no mover in this scene / --no_mover", flush=True)
    else:
        for k in range(int(6.0 / dt)):
            mover_step("TEST", force=(k == 0))
            legs_term.cmd[:] = torch.tensor([0.0, 0.0, args.wz_bias], device=dev)
            obs, _, _, _, _ = env.step(arm_action(ready, True))
            if k % int(1.0 / dt) == 0:
                mp = T(mover.data.root_pos_w)[0]
                print(f"[MOVER] test t={k * dt:.0f}s commanded ({MV['xy'][0]:+.2f},{MV['xy'][1]:+.2f}) sim ({float(mp[0] - origin[0]):+.2f},{float(mp[1] - origin[1]):+.2f}) z={float(mp[2]):.2f}", flush=True)
        lx, ly = float(MV["xy"][0]) + 2.0, float(MV["xy"][1])
        r = link.scan(lx, ly, math.pi)                     # look west at the box from 2 m east
        ahead = float(r[len(r) // 2])                      # ray at angle 0 = straight ahead
        print(f"[MOVER] test: lidar from 2 m east, straight ahead {ahead:.2f} m (box face expected at {2.0 - W.MOVER['size'][0] / 2:.2f} m) "
              f"-> {'SEES THE BOX' if abs(ahead - (2.0 - W.MOVER['size'][0] / 2)) < 0.05 else 'DOES NOT SEE THE BOX'}", flush=True)
    print("[MOVER] test done - quitting", flush=True)
    env.close(); app.close(); sys.exit(0)
def scan_report(tag):
    p, yaw = base_pose()
    rx, ry = rel(p)
    t0 = time.time(); rp = _scan(rx, ry, yaw); ms = (time.time() - t0) * 1000
    rb = _box_scan(rx, ry, yaw)
    both = np.isfinite(rp) & np.isfinite(rb)
    diff = np.abs(rp[both] - rb[both])
    print(f"[SCAN] {tag} at ({rx:+.2f},{ry:+.2f}) yaw {math.degrees(yaw):+.0f}: {ms:.1f} ms/scan | hits {int(np.isfinite(rp).sum())}/{len(rp)} "
          f"| min range {float(np.min(rp)):.2f} m | vs box list: median diff {float(np.median(diff)) if diff.size else float('nan'):.3f} m, "
          f"{int((diff > 0.10).sum())} rays differ >10 cm", flush=True)
    return rp, rb
if args.scan_test:
    for _ in range(int(1.0 / dt)):
        legs_term.cmd[:] = torch.tensor([0.0, 0.0, 0.0], device=dev)
        obs, _, _, _, _ = env.step(arm_action(ready, True))
    rp, rb = scan_report("TEST")
    n = len(rp)
    for k in range(8):
        sl = slice(k * n // 8, (k + 1) * n // 8)
        print(f"[SCAN]   sector {-180 + 45 * k:+4d}..{-135 + 45 * k:+4d} deg: physx min {float(np.min(rp[sl])):5.2f} m | box min {float(np.min(rb[sl])):5.2f} m", flush=True)
    if args.scan == "physx":
        print(f"[SCAN]   prims hit: {dict(sorted(_scan.hit_paths.items(), key=lambda kv: -kv[1]))}", flush=True)
        print(f"[SCAN]   raw PhysX hits {_scan.raw_hits} (on Spot itself {_scan.robot_hits})", flush=True)
    print("[SCAN] test done - quitting", flush=True)
    env.close(); app.close(); sys.exit(0)
enter("FOLD" if args.auto else "WAIT", hold_xy=p0, hold_yaw=0.0)
if not args.auto:
    print("[DEMO] READY: start your recording, then click this terminal and press g", flush=True)
step = 0
try:
    while app.is_running():
        with torch.inference_mode():
            nav_cmd = link.recv_cmd(step)
            S, t = state, state["t"]
            cube = T(obj.data.root_pos_w)[0, :3]
            ee_p = T(ee.data.target_pos_w)[0, 0, :3]
            f1x = float(T(robot.data.joint_pos)[0, f_id])
            in_grip = float(torch.norm(cube - ee_p)) < 0.06 and -0.9 < f1x < -0.2
            ph = S["phase"]
            S["info"] = ""
            legs = hold_cmd(S["hold_xy"], S["hold_yaw"]) if S["hold_xy"] is not None else [0.0, 0.0, 0.0]
            if ph == "WAIT":
                a = arm_action(ready, True)
                if start[0]:
                    enter("FOLD")
            elif ph == "BODYLOOK":                                   # BODY CAM: stand still, one look at the table
                a = arm_action(stow, True)
                if t == int(0.6 / dt):
                    _ck = cam; cam = BODY_CAM; save_cam_image("bodycam"); cam = _ck   # image only, detection below
                    try:                                                              # BODYCAM DEBUG: camera pose vs base
                        _cp = T(BODY_CAM.data.pos_w)[0]; _cq = T(BODY_CAM.data.quat_w_world)[0:1]
                        _fw = quat_apply(_cq, torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
                        _bp, _by = base_pose()
                        print(f"[BODYCAM] camera at map ({float(_cp[0] - origin[0]):+.2f},{float(_cp[1] - origin[1]):+.2f}) z {float(_cp[2]):.2f}"
                              f" looking ({float(_fw[0]):+.2f},{float(_fw[1]):+.2f},{float(_fw[2]):+.2f}) | base ({float(_bp[0] - origin[0]):+.2f},"
                              f"{float(_bp[1] - origin[1]):+.2f}) yaw {math.degrees(_by):+.0f}", flush=True)
                    except Exception as _e:
                        print(f"[BODYCAM] pose debug failed: {_e}", flush=True)
                    _scan_snap()
                    enter("ASK")
            elif ph == "SCAN":                                       # CAMERA: arm to the next scan pose
                a = arm_move(SCAN_POSES[FLOW["scan_i"]], True, "SCAN_SNAP")
            elif ph == "SCAN_SNAP":                                  # CAMERA: hold still, then detect the cubes in view
                a = arm_action(SCAN_POSES[FLOW["scan_i"]], True)
                if t == int(0.6 / dt):
                    _scan_snap()
                    FLOW["scan_i"] += 1
                    enter("SCAN" if FLOW["scan_i"] < len(SCAN_POSES) else "SCAN_FOLD")
            elif ph == "SCAN_FOLD":                                  # CAMERA: arm back to stow, then ask
                a = arm_move(stow, True, "ASK")
            elif ph == "ASK":                                        # FLOW: standing at the table, waiting for a colour
                a = arm_action(stow, True)
                _ask_tick(t)
            elif ph == "BACK_TABLE":                                 # FLOW: straight back 0.5 m from the bench (clearance to sidestep)
                a, legs = drive(VIEW[0], PICK_YAW, stow, True, lambda: enter("SHIFT"), timeout=6.0)
            elif ph in ("SHIFT", "SHIFT_WALK"):                       # FLOW: turn to the new pre-pose, then walk forward to it
                _d = prepick - VIEW[0]
                _wy = math.atan2(float(_d[1]), float(_d[0])) if float(torch.norm(_d)) > 0.05 else PICK_YAW   # same pre-pose: no turn
                if float(torch.norm(_d)) <= 0.05:                    # same pre-pose: no shift needed, go straight in
                    a, legs = arm_action(stow, True), [0.0, 0.0, 0.0]
                    enter("APPROACH")
                elif ph == "SHIFT":
                    a, legs = drive(VIEW[0], _wy, stow, True, lambda: enter("SHIFT_WALK"), timeout=8.0)
                else:                                            # walk straight; stop within 2 cm ALONG the walk
                    _p, _yw = base_pose()
                    _e = prepick - _p
                    _al = float(_e[0]) * math.cos(_wy) + float(_e[1]) * math.sin(_wy)
                    _dy = wrap(_wy - _yw)
                    _wz = max(-0.6, min(0.6, 1.5 * _dy)) if abs(_dy) > YAW_TOL * 0.5 else 0.0
                    _vx = 0.0 if abs(_al) < 0.015 else math.copysign(max(0.15, min(0.4, 1.2 * abs(_al))), _al)
                    legs = [_vx, 0.0, _wz]
                    a = arm_action(stow, True)
                    S["info"] = f"walk along={_al:+.3f} dyaw={math.degrees(_dy):+.0f}"
                    S["good"] = S["good"] + 1 if abs(_al) < 0.02 and abs(_dy) < YAW_TOL else 0
                    if S["good"] >= HOLD or t * dt > 10.0:
                        _cr = -float(_e[0]) * math.sin(_wy) + float(_e[1]) * math.cos(_wy)
                        print(f"[MULTI] shift walk done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across"
                              f"{' (timeout)' if S['good'] < HOLD else ''}", flush=True)
                        legs = [0.0, 0.0, 0.0]
                        enter("SHIFT_BACK")
            elif ph == "SHIFT_BACK":                                 # FLOW: turn back to face the table
                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=8.0)
            elif ph == "REAPPROACH":                                 # FLOW: parked badly - back to the pre-pose, then APPROACH again
                a, legs = drive(prepick, PICK_YAW, stow, True, lambda: enter("APPROACH"), timeout=8.0)
            elif ph == "BACKOFF_BIN":                                # FLOW: step back from the bin to the pre-place pose
                a, legs = drive(preplace, PLACE_YAW, stow, True, lambda: enter("TURN_TABLE"), timeout=6.0)
            elif ph == "TURN_TABLE":                                 # FLOW: scripted turn to face the table before Nav2
                _d = VIEW[0] - preplace
                a, legs = drive(preplace, math.atan2(float(_d[1]), float(_d[0])), stow, True, lambda: enter("NAV_TABLE"), timeout=8.0)
            elif ph == "NAV_TABLE":                                  # FLOW: back from the bin to the table view pose
                a = arm_action(stow, True)
                legs = nav_phase(prepick, PICK_YAW, lambda: enter("APPROACH"), nav_cmd)
            elif ph == "IDLE":                                       # PANEL: standing, arm stowed, waiting for a button
                a = arm_action(stow, True)
            elif ph == "FOLD_FREE":
                a = arm_move(stow, True, "NAV_FREE")
            elif ph == "NAV_FREE":
                a = arm_action(stow, True)
                legs = nav_phase(PANEL["pre"], PANEL["yaw"], _free_after_nav, nav_cmd)
            elif ph == "FINE_FREE":
                a, legs = drive(PANEL["stand"], PANEL["yaw"], stow, True, _free_done, timeout=15.0)
            elif ph == "FOLD":
                a = arm_move(stow, True, "NAV_PICK")
            elif ph == "NAV_PICK":
                a = arm_action(stow, True)
                legs = nav_phase(prepick, PICK_YAW, lambda: enter("APPROACH"), nav_cmd)
            elif ph == "APPROACH":
                if args.body_cam and args.multi_flow and SEL["colour"] is None:      # BODY CAM: look from the pre-pose
                    a, legs = drive(prepick, PICK_YAW, stow, True,
                                    lambda: (DET.clear(), FLOW.update(scan_i=0), enter("BODYLOOK", hold_xy=prepick, hold_yaw=PICK_YAW)))
                else:
                    _p, _yw = base_pose()                              # straight in, stop within 3 cm along the heading
                    _e = bench_park - _p
                    _al = float(_e[0]) * math.cos(PICK_YAW) + float(_e[1]) * math.sin(PICK_YAW)
                    _cr = -float(_e[0]) * math.sin(PICK_YAW) + float(_e[1]) * math.cos(PICK_YAW)
                    _hd = PICK_YAW
                    if _al > 0.10 and abs(_cr) > 0.015:               # steer toward the stand pose while far out
                        _hd = PICK_YAW + max(-0.25, min(0.25, math.atan2(_cr, _al)))
                    _dy = wrap(_hd - _yw)
                    _wz = 0.0 if abs(_dy) < 0.015 else (math.copysign(max(0.30, min(0.6, 1.5 * abs(_dy))), _dy)
                                                       - (0.0 if args.no_script_bias else args.wz_bias))   # NET turn >= 0.30 after the bias
                    _vx = 0.0 if abs(_al) < 0.025 else math.copysign(max(0.2, min(0.4, 1.2 * abs(_al))), _al)
                    _vy = 0.0
                    legs = [_vx, _vy, _wz]
                    a = arm_action(stow, True)
                    S["info"] = f"approach along={_al:+.3f} across={_cr:+.3f} dyaw={math.degrees(_dy):+.0f}"
                    _yf = wrap(PICK_YAW - _yw)
                    S["good"] = S["good"] + 1 if abs(_al) < 0.03 and abs(_yf) < 0.03 else 0
                    if S["good"] >= HOLD or t * dt > 15.0:
                        print(f"[MULTI] approach done: {_al * 100:+.1f} cm along, {_cr * 100:+.1f} cm across, heading {math.degrees(_yf):+.1f} deg"
                              f"{' (timeout)' if S['good'] < HOLD else ''}", flush=True)
                        legs = [0.0, 0.0, 0.0]
                        _after_approach()
            elif ph == "UNFOLD":
                a = arm_move(ready, True, "SETTLE")
            elif ph == "SETTLE":
                a = arm_action(ready, True)
                if args.settle_train_gains and t <= 1:
                    apply_gains("PICK")                                  # let the arm sag/settle before PICK
                _p, _yw = base_pose()                                    # undo the UNFOLD shove: heading + forward only
                _dd = S["hold_xy"] - _p
                _fw, _lat = to_body(float(_dd[0]), float(_dd[1]), _yw)
                _sy = wrap(S["hold_yaw"] - _yw)
                _vx = math.copysign(max(args.repark_vmin, min(0.3, 2.0 * abs(_fw))), _fw) if abs(_fw) > 0.02 else 0.0
                _wz = (math.copysign(max(args.hold_wz_min, min(0.4, 1.5 * abs(_sy))), _sy) - _wz_bias()) if abs(_sy) > 0.025 else 0.0
                legs = [_vx, 0.0, _wz]
                S["info"] = f"settle hold fwd={_fw:+.3f} lat={_lat:+.3f} dyaw={math.degrees(_sy):+.1f}"
                _ad = (arm_q() - ready).abs()
                _ae = float(_ad.max()); _av = float(T(robot.data.joint_vel)[0, arm_ids].abs().max())
                _arm_ok = (not args.settle_tight) or (_ae < args.settle_arm_tol and _av < 0.05)
                if t >= int(0.7 / dt) and ((abs(_fw) <= 0.02 and abs(_sy) <= 0.025 and _arm_ok) or t * dt > 4.0):
                    print(f"[DIAG] settle end: arm err {_ae:.3f} rad (worst {arm_names[int(_ad.argmax())][5:]}), "
                          f"arm vel {_av:.3f}, t {t * dt:.1f} s", flush=True)
                    print(f"[PERC] settle hold {'done' if t * dt <= 3.95 else 'timeout'}: fwd {_fw:+.3f} m, "
                          f"lat {_lat:+.3f} m (not corrected), yaw {math.degrees(_sy):+.1f} deg", flush=True)
                    S["held"] = 0
                    enter("LOOK")
            elif ph == "LOOK":
                if args.settle_train_gains and t <= 1:
                    apply_gains("PICK")
                a = arm_action(LOOK_POSES[P["look_i"]], True)
                perc_look_step(t)
            elif ph == "GIVEUP":                                    # open (unless holding), arm to stow, base held
                a = arm_move(stow, not state.get("giveup_hold", False), "GIVEUP_DONE", move_steps=MOVE_STEPS)
            elif ph == "GIVEUP_DONE":
                a = arm_action(stow, not state.get("giveup_hold", False))
                after_giveup()
            elif ph == "REPARK":
                a = arm_action(ready, True)
                legs, _dist = repark_cmd(S["hold_xy"], S["hold_yaw"])
                S["info"] = f"repark dist={_dist:.3f}"
                S["good"] = S["good"] + 1 if _dist <= 0.02 else 0
                if S["good"] >= HOLD or t * dt > 6.0:
                    if S["good"] < HOLD:
                        print(f"[PERC] re-park timeout, {_dist:.3f} m left - looking again anyway", flush=True)
                    P["look_i"] = 0
                    enter("LOOK")
            elif ph == "LOOKMOVE":
                a = arm_move(LOOK_POSES[P["look_i"]], True, "LOOK")
            elif ph == "UNLOOK":
                if args.settle_train_gains and t <= 1:
                    apply_gains("PICK")                                  # no kp switch right before PICK
                a = arm_move(ready, True, "PICK")
            elif ph == "RETRY":
                a = arm_move(ready, True, "LOOK")
            elif ph == "PICK":
                legs = hold_cmd(S["hold_xy"], S["hold_yaw"], xy=False)      # heading only, closest to training
                if S["t"] <= 1:
                    _d = arm_q() - ready
                    _bd = robot.data
                    print(f"[DIAG] PICK start: yaw rate {float(T(_bd.root_ang_vel_b)[0, 2]):+.3f} rad/s, "
                          f"side {float(T(_bd.root_lin_vel_b)[0, 1]):+.3f} fwd {float(T(_bd.root_lin_vel_b)[0, 0]):+.3f} m/s | "
                          f"arm err max {float(_d.abs().max()):.3f} ("
                          + " ".join(f"{n[5:]}:{float(e):+.3f}" for n, e in zip(arm_names, _d))
                          + f") | arm vel {float(T(_bd.joint_vel)[0, arm_ids].abs().max()):.3f}", flush=True)
                a, ok, fail = perc_pick_step(obs)
                if args.pick_legs_zero:
                    legs = [0.0, 0.0, 0.0]
                S["held"] = S["held"] + 1 if ok else 0
                if fail:
                    S["held"] = 0
                    retry_or_stop(P["why"])
                if S["held"] >= HOLD:
                    pp, py = base_pose()
                    _px, _d, _r = P.get("last", (0, float("nan"), 0.0))
                    print(f"[PERC] pick verified: cube {_px} px at {_d:.2f} m in hand camera, lifted {_r:.2f} m", flush=True)
                    enter("STOW", hold_xy=pp, hold_yaw=py)
                elif t * dt > args.pick_timeout and S["phase"] == "PICK":
                    retry_or_stop(f"pick not completed in {args.pick_timeout:.0f} s")
            elif ph == "STOW":
                a = arm_move(stow, False, "BACKOFF")
                if S["phase"] == "BACKOFF":
                    _cb = subtract_frame_transforms(T(robot.data.root_pos_w), T(robot.data.root_quat_w), cube.unsqueeze(0))[0][0]
                    print(f"[DEMO] stowed | {arm_diag(stow)} | cube vs base (log) x {float(_cb[0]):+.3f} "
                          f"y {float(_cb[1]):+.3f} z {float(_cb[2]):+.3f} m", flush=True)
                    _ok, _px, _dm = hand_check()
                    if _ok:
                        print(f"[PERC] after STOW: cube still in hand ({_px} px at {_dm:.2f} m)", flush=True)
                    else:
                        retry_or_stop(f"cube lost during STOW (hand camera {_px} px, depth {_dm:.2f} m)")
                    pp, _ = base_pose()
                    S["bk"] = prepick.clone()                              # GOTO: back off to the checked pre-pose
            elif ph == "BACKOFF":
                a, legs = drive(S["bk"], PICK_YAW, stow, False, lambda: enter("NAV_BIN"), timeout=6.0)
                _rp = min(1.0, t * dt / 1.0)                      # CARRY: gentle start, max 0.2 m/s while holding the cube
                legs = [max(-0.2, min(0.2, legs[0])) * _rp, max(-0.2, min(0.2, legs[1])) * _rp, legs[2]]
            elif ph == "NAV_BIN":
                a = arm_action(stow, False)
                legs = nav_phase(preplace, PLACE_YAW, _to_go_bin, nav_cmd)
            elif ph == "GO_BIN":
                tgt, yaw_t = S["wp"][S["wp_i"]]
                def nxt():
                    S["wp_i"] += 1
                    S["good"], S["t"] = 0, -1
                    if S["wp_i"] >= len(S["wp"]):
                        enter("PLACE", hold_xy=bin_park, hold_yaw=PLACE_YAW)
                a, legs = drive(tgt, yaw_t, stow, False, nxt, timeout=6.0 if S["wp_i"] == 0 else 15.0)
            elif ph == "PLACE":
                if not state.get("via_done"):                       # stow -> midpoint first, slowly
                    _t0 = S["t"]
                    a = arm_move(VIA, False, "PLACE", on_stuck=place_recover, move_steps=PLACE_STEPS, wrist_tol=args.place_wrist_tol)
                    if S["phase"] == "PLACE" and S["t"] == 0 and _t0 != 0:
                        state["via_done"] = True
                        print("[PLACE] midpoint reached, moving to place pose", flush=True)
                else:
                    a = arm_move(place, False, "RELEASE", on_stuck=place_recover,
                                 move_steps=PLACE_STEPS * (1 + state.get("place_tries", 0)), wrist_tol=args.place_wrist_tol)
                if S["phase"] == "RELEASE":
                    off = ee_p[:2] - bin_w
                    print(f"[DEMO] gripper vs bin centre: dx={float(off[0]):+.3f} dy={float(off[1]):+.3f} m "
                          f"| gripper height {float(ee_p[2]):.3f}, bin top {BIN_TOP:.3f}", flush=True)
                    coff = carried_cube_w()[:2] - bin_w
                    print(f"[PLACE] carried cube vs bin centre (perception) dx {float(coff[0]):+.3f} dy {float(coff[1]):+.3f} m", flush=True)
                    if (abs(float(coff[0])) > args.place_tol or abs(float(coff[1])) > args.place_tol) and state.get("place_fixes", 0) < 1:
                        state["place_fixes"] = state.get("place_fixes", 0) + 1
                        pp, pyaw = base_pose()
                        fwd, lat = to_body(-float(coff[0]), -float(coff[1]), pyaw)     # cube move needed, robot frame
                        fwd_c = max(-args.place_fix_max, min(args.place_fix_max, fwd))
                        r = float(torch.norm(carried_cube_w()[:2] - shoulder_xy()))
                        dq = max(-0.25, min(0.25, args.sh0_sign * lat / max(r, 0.3)))
                        place[_sh0] += dq                                              # sideways: shoulder turn
                        print(f"[PLACE] fix step dx {fwd_c:+.3f} dy +0.000 m (robot frame, base; needed fwd {fwd:+.3f}"
                              f"{', capped' if abs(fwd) > args.place_fix_max else ''}) | arm sideways {lat:+.3f} m via sh0 {dq:+.3f} rad (r {r:.2f} m)", flush=True)
                        if abs(fwd_c) > 0.02:
                            enter("PLACE_FIX", hold_xy=pp + torch.tensor([math.cos(pyaw) * fwd_c, math.sin(pyaw) * fwd_c], device=dev), hold_yaw=pyaw)
                        else:
                            enter("PLACE")
            elif ph == "PLACE_FIX":                                 # one small step, no turning, then release
                a = arm_action(place, False)
                legs, _dist = repark_cmd(S["hold_xy"], S["hold_yaw"])
                legs[2] = 0.0
                S["info"] = f"place_fix dist={_dist:.3f}"
                S["good"] = S["good"] + 1 if _dist <= 0.02 else 0
                if S["good"] >= HOLD or t * dt > 4.0:
                    enter("PLACE")
            elif ph == "PLACE_VIA":
                a = arm_move(VIA, False, "PLACE", move_steps=PLACE_STEPS, wrist_tol=args.place_wrist_tol)
            elif ph == "RELEASE":
                a = arm_action(place, True)
                if t >= int(1.5 / dt):
                    enter("RETRACT")
            elif ph == "RETRACT":
                a = arm_move(stow, True, "DONE")
                if S["phase"] == "DONE":
                    d = cube[:2] - bin_w
                    inside = abs(float(d[0])) < BIN_HALF[0] and abs(float(d[1])) < BIN_HALF[1] and float(cube[2]) < BIN_TOP + 0.02
                    print(f"[RESULT] {'CUBE IN BIN' if inside else 'MISSED'}: cube vs bin centre dx={float(d[0]):+.3f} "
                          f"dy={float(d[1]):+.3f} z={float(cube[2]):.3f} m", flush=True)
                    for _cn, (_x0, _y0) in ({} if args.multi_flow else DISTRACT).items():   # MULTI: did the pick disturb the neighbours?
                        _q = T(u.scene[f"cube_{_cn}"].data.root_pos_w)[0]
                        _mv = math.hypot(float(_q[0] - origin[0]) - _x0, float(_q[1] - origin[1]) - _y0)
                        print(f"[MULTI] neighbour {_cn}: moved {_mv * 100:.1f} cm, z={float(_q[2]):.3f} -> {'UNTOUCHED' if _mv < 0.02 and float(_q[2]) > BENCH_H + 0.015 else 'DISTURBED'}", flush=True)
                    if args.multi_flow:
                        _next_round()
            else:
                a = arm_action(stow, True)
                if args.exit_on_done and ph == "DONE" and t * dt > 1.0:
                    break
            mover_step(ph)
            panel_tick(ph)
            if ph not in ("NAV_PICK", "NAV_BIN", "NAV_FREE", "NAV_TABLE", "PICK") and not args.no_script_bias:
                legs = [legs[0], legs[1], legs[2] + args.wz_bias]     # walking policy yaw-rate offset (same fix as Nav2 phases)
            legs_term.cmd[:] = torch.tensor(legs, device=dev)
            obs, _, _, _, _ = env.step(a)
            _pw = T(robot.data.root_pos_w)[0]
            _, _, _yw = euler_xyz_from_quat(T(robot.data.root_quat_w))
            _lv, _av = T(robot.data.root_lin_vel_b)[0], T(robot.data.root_ang_vel_b)[0]
            _gx, _gy, _gyaw = float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_yw[0])
            _vx, _vy, _wz = float(_lv[0]), float(_lv[1]), float(_av[2])
            _ox, _oy, _oyaw = (_gx, _gy, _gyaw) if args.no_drift else odom.step(_gx, _gy, _gyaw, _vx, _vy, _wz)
            link.send(step, _ox, _oy, float(_pw[2]), _oyaw, _vx, _vy, _wz, scan_pose=(_gx, _gy, _gyaw), gt=(_gx, _gy, _gyaw))
            if (not args.no_drift) and step % int(10.0 / dt) == 0:
                print(f"[ODOM] t={step * dt:.0f}s true ({_gx:+.2f},{_gy:+.2f},{math.degrees(_gyaw):+.0f}) odom ({_ox:+.2f},{_oy:+.2f},{math.degrees(_oyaw):+.0f}) "
                      f"drift {math.hypot(_ox - _gx, _oy - _gy) * 100:.0f} cm {math.degrees(math.atan2(math.sin(_oyaw - _gyaw), math.cos(_oyaw - _gyaw))):+.1f} deg", flush=True)
            if args.scan == "physx" and step % int(10.0 / dt) == 0:
                scan_report(f"t={step * dt:.0f}s")
            if step % int(0.5 / dt) == 0:
                p, yaw = base_pose()
                g = T(robot.data.projected_gravity_b)[0]
                tilt = float(torch.rad2deg(torch.acos(torch.clamp(-g[2], -1.0, 1.0))))
                print(f"[STATUS] {step * dt:6.1f}s {S['phase']:8s} base=({float(p[0] - origin[0]):+.2f},{float(p[1] - origin[1]):+.2f}) "
                      f"yaw={math.degrees(yaw):+5.0f} cube_z={float(cube[2]):.3f} grip={in_grip} tilt={tilt:4.1f} {S['info']}", flush=True)
        state["t"] += 1
        step += 1
finally:
    env.close()
    app.close()
