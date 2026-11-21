"""FLOW + BODY CAMERA: a front camera on Spot's body (like the real Spot's front cameras) identifies the cubes on the table
in one look; the wrist camera still does the precise pick. With --body_cam: at the table Spot stands still, the body camera
runs perception's detect_cubes() (colour from RGB, position from depth), Spot asks, and the stand pose is planned from the
body-camera position. Same 640x480 image size as the hand camera (detect_cubes' pixel grid). Saves one body-camera image
per look to @SPOT_DEMO_ROOT@/logs/perception/demo_frames/ (check the aim). Needs fix_flow_camdetect.py. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "BODY CAM:" in s:
    print("[FIX] body camera already there"); sys.exit(0)
if "CAMERA: entity key" not in s:
    print("[FIX] body camera: needs fix_flow_camdetect.py first - skipped"); sys.exit(0)
R = [
 ('parser.add_argument("--order"',
  'parser.add_argument("--body_cam", action="store_true", help="FLOW: identify the cubes with a front BODY camera (wrist camera still picks)")\n'
  'parser.add_argument("--body_cam_link", type=str, default="base", help="BODY CAM: robot link the camera is mounted on")\n'
  'parser.add_argument("--body_cam_pos", type=float, nargs=3, default=[0.47, 0.0, 0.0], help="BODY CAM: mount position in the link frame (m)")\n'
  'parser.add_argument("--body_cam_pitch", type=float, default=20.0, help="BODY CAM: tilt down (deg)")\n'
  'parser.add_argument("--body_cam_hfov", type=float, default=90.0, help="BODY CAM: horizontal field of view (deg)")\n'
  'parser.add_argument("--order"'),
 ('cfg.scene.hand_cam.update_latest_camera_pose = True\n',
  'cfg.scene.hand_cam.update_latest_camera_pose = True\n'
  'if args.body_cam:                                  # BODY CAM: front camera on the body, looking forward and down\n'
  '    _bp = math.radians(args.body_cam_pitch)\n'
  '    from isaaclab.utils.math import quat_from_euler_xyz as _qe          # quaternion in Isaac Lab\'s own order (3.0: x,y,z,w)\n'
  '    _bq = tuple(float(v) for v in _qe(torch.tensor([0.0]), torch.tensor([_bp]), torch.tensor([0.0]))[0])\n'
  '    cfg.scene.body_cam = CameraCfg(\n'
  '        prim_path=f"{cfg.scene.robot.prim_path}/{args.body_cam_link}/body_cam",\n'
  '        update_period=0.0, height=480, width=640,\n'
  '        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],\n'
  '        colorize_instance_id_segmentation=False,\n'
  '        spawn=sim_utils.PinholeCameraCfg(focal_length=10.0, horizontal_aperture=20.0 * math.tan(math.radians(args.body_cam_hfov) / 2),\n'
  '                                         clipping_range=(0.05, 6.0)),\n'
  '        offset=CameraCfg.OffsetCfg(pos=tuple(args.body_cam_pos), rot=_bq, convention="world"),\n'
  '    )\n'
  '    cfg.scene.body_cam.update_latest_camera_pose = True\n'
  '    print(f"[BODYCAM] front body camera on {args.body_cam_link} at {args.body_cam_pos}, {args.body_cam_pitch:.0f} deg down, {args.body_cam_hfov:.0f} deg wide", flush=True)\n'),
 ('cam = u.scene["hand_cam"]\n',
  'cam = u.scene["hand_cam"]\n'
  'BODY_CAM = u.scene["body_cam"] if args.body_cam else None     # BODY CAM handle\n'),
 ('    for name, xy in detect_cubes().items():\n',
  '    for name, xy in (_detect_with(BODY_CAM) if args.body_cam else detect_cubes()).items():\n'),
 ('        DET.setdefault(best, {"seen": [], "colour": name.split("_")[0]})["seen"].append(xy)\n',
  '        DET.setdefault(best, {"seen": [], "colour": name.split("_")[0]})["seen"].append(xy)\n'
  '        print(f"[DETECT] {name} -> {best}: camera vs truth (log only) {bd * 100:.1f} cm", flush=True)\n'),
 ('def _det_xy(k):\n',
  'def _detect_with(c, image_tag=None):\n'
  '    """BODY CAM: run perception\'s detect_cubes() / save_cam_image() on another camera (they read the global cam)."""\n'
  '    global cam\n'
  '    _keep = cam\n'
  '    cam = c\n'
  '    try:\n'
  '        if image_tag:\n'
  '            save_cam_image(image_tag)\n'
  '        return detect_cubes()\n'
  '    finally:\n'
  '        cam = _keep\n'
  'def _det_xy(k):\n'),
 ('        enter("SCAN" if args.cam_detect else "ASK", hold_xy=bench_park, hold_yaw=PICK_YAW)\n',
  '        enter("BODYLOOK" if args.body_cam else ("SCAN" if args.cam_detect else "ASK"), hold_xy=bench_park, hold_yaw=PICK_YAW)\n'),
 ('        if args.cam_detect:\n',
  '        if args.cam_detect or args.body_cam:\n'),
 ('hand camera, {len(SCAN_POSES)} scan poses',
  "{'body camera' if args.body_cam else f'hand camera, {len(SCAN_POSES)} scan poses'}"),
 ('    if args.cam_detect and colour in DET:',
  '    if (args.cam_detect or args.body_cam) and colour in DET:'),
 ('            elif ph == "SCAN":',
  '            elif ph == "BODYLOOK":                                   # BODY CAM: stand still, one look at the table\n'
  '                a = arm_action(stow, True)\n'
  '                if t == int(0.6 / dt):\n'
  '                    _detect_with(BODY_CAM, image_tag="bodycam")\n'
  '                    _scan_snap()\n'
  '                    enter("ASK")\n'
  '            elif ph == "SCAN":'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] body camera (--body_cam): front camera identifies the cubes, wrist camera picks")
