"""FLOW + CAMERA: with --cam_detect the cube list at the table comes from the hand camera (perception's detect_cubes()),
not from the sim. At the ASK pose the arm unfolds and pans left / centre / right, detect_cubes() runs at each pose,
detections are merged, the arm folds back, and Spot asks. The chosen cube's stand pose is planned from the CAMERA position.
Answers can be the camera colour word ("red") or the cube name ("dex"). Known sim shortcut (same as perception's):
object masks come from instance segmentation; a detection is tied to its sim object by nearest position (log/mask only).
Needs perception v8 (detect_cubes). Without --cam_detect nothing changes. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "CAMERA: entity key" in s:
    print("[FIX] camera cube detection already there"); sys.exit(0)
if "def detect_cubes" not in s:
    print("[FIX] camera detection: perception file has no detect_cubes() - skipped"); sys.exit(0)
R = [
 ('parser.add_argument("--order"',
  'parser.add_argument("--cam_detect", action="store_true", help="FLOW: list the cubes with the hand camera (detect_cubes) instead of the sim")\n'
  'parser.add_argument("--scan_pan", type=float, default=0.5, help="FLOW: shoulder pan (rad) for the left/right camera scan poses")\n'
  'parser.add_argument("--order"'),
 ('# ---------------- PANEL: operator panel (nav side) ----------------\n',
  'DET = {}                                       # CAMERA: entity key -> {"seen": [(x, y), ...], "colour": camera colour word}\n'
  'SCAN_POSES = [_pan(args.scan_pan), ready.clone(), _pan(-args.scan_pan)]\n'
  'def _scan_snap():\n'
  '    for name, xy in detect_cubes().items():\n'
  '        best, bd = None, 0.06\n'
  '        for k in REMAIN:                       # tie the detection to its sim object (mask/log only), nearest within 6 cm\n'
  '            _q = T(u.scene[CUBES[k][0]].data.root_pos_w)[0]\n'
  '            _d = math.hypot(float(_q[0] - origin[0]) - xy[0], float(_q[1] - origin[1]) - xy[1])\n'
  '            if _d < bd:\n'
  '                best, bd = k, _d\n'
  '        if best is None:\n'
  '            print(f"[DETECT] {name} at ({xy[0]:+.3f},{xy[1]:+.3f}) is not one of the cubes left - ignored", flush=True)\n'
  '            continue\n'
  '        DET.setdefault(best, {"seen": [], "colour": name.split("_")[0]})["seen"].append(xy)\n'
  'def _det_xy(k):\n'
  '    v = sorted(DET[k]["seen"]); n = len(v)\n'
  '    return (sorted(a for a, _ in v)[n // 2], sorted(b for _, b in v)[n // 2])\n'
  '# ---------------- PANEL: operator panel (nav side) ----------------\n'),
 ('        enter("ASK", hold_xy=bench_park, hold_yaw=PICK_YAW)\n',
  '        DET.clear(); FLOW["scan_i"] = 0\n'
  '        enter("SCAN" if args.cam_detect else "ASK", hold_xy=bench_park, hold_yaw=PICK_YAW)\n'),
 ('        seen = ", ".join(f"{c} at ({CUBES[c][2][0]:+.2f},{CUBES[c][2][1]:+.2f})" for c in REMAIN)\n',
  '        if args.cam_detect:\n'
  '            seen = ", ".join((f"{DET[c][\'colour\']} ({c}) at ({_det_xy(c)[0]:+.2f},{_det_xy(c)[1]:+.2f})" if c in DET else f"{c}: NOT SEEN")\n'
  '                             for c in REMAIN)\n'
  '            print(f"[MULTI] cubes on the table (hand camera, {len(SCAN_POSES)} scan poses): {seen}", flush=True)\n'
  '        seen = ", ".join(f"{c} at ({CUBES[c][2][0]:+.2f},{CUBES[c][2][1]:+.2f})" for c in REMAIN)\n'),
 ('    colour = _parts[0] if _parts else colour\n',
  '    colour = _parts[0] if _parts else colour\n'
  '    if colour not in REMAIN:                   # CAMERA: accept the camera colour word too ("green" -> dex)\n'
  '        colour = next((k for k in REMAIN if k in DET and DET[k]["colour"] == colour), colour)\n'),
 ('    ent, prim, xy = CUBES[colour]\n',
  '    ent, prim, xy = CUBES[colour]\n'
  '    if args.cam_detect and colour in DET:      # CAMERA: plan the stand pose from where the camera saw the cube\n'
  '        xy = _det_xy(colour)\n'
  '        print(f"[MULTI] {colour}: planning from the camera position ({xy[0]:+.3f},{xy[1]:+.3f})", flush=True)\n'),
 ('            elif ph == "ASK":',
  '            elif ph == "SCAN":                                       # CAMERA: arm to the next scan pose\n'
  '                a = arm_move(SCAN_POSES[FLOW["scan_i"]], True, "SCAN_SNAP")\n'
  '            elif ph == "SCAN_SNAP":                                  # CAMERA: hold still, then detect the cubes in view\n'
  '                a = arm_action(SCAN_POSES[FLOW["scan_i"]], True)\n'
  '                if t == int(0.6 / dt):\n'
  '                    _scan_snap()\n'
  '                    FLOW["scan_i"] += 1\n'
  '                    enter("SCAN" if FLOW["scan_i"] < len(SCAN_POSES) else "SCAN_FOLD")\n'
  '            elif ph == "SCAN_FOLD":                                  # CAMERA: arm back to stow, then ask\n'
  '                a = arm_move(stow, True, "ASK")\n'
  '            elif ph == "ASK":'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] camera cube detection (--cam_detect): scan left/centre/right at the table, plan from camera positions")
