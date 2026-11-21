"""FLOW: (1) body camera debug - at every body-camera look print where the camera is and where it points (map frame),
so a blank image can be traced (pose not updated vs. something in front). (2) An unknown place name (e.g. --order red:bin2
in a scene without bin2) falls back to 'bin' with a warning instead of leaving the flow waiting forever. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "BODYCAM DEBUG" in s:
    print("[FIX] body camera debug already there"); sys.exit(0)
R = [
 ('''        _say(f"no place called '{_tgt}' - choose from {list(PLACES)}"); return\n''',
  '''        print(f"[MULTI] WARNING: no place called '{_tgt}' in this scene ({list(PLACES)}) - using bin", flush=True)\n'''
  '''        _tgt = "bin"\n'''),
 ('                    _ck = cam; cam = BODY_CAM; save_cam_image("bodycam"); cam = _ck   # image only, detection below\n',
  '                    _ck = cam; cam = BODY_CAM; save_cam_image("bodycam"); cam = _ck   # image only, detection below\n'
  '                    try:                                                              # BODYCAM DEBUG: camera pose vs base\n'
  '                        _cp = T(BODY_CAM.data.pos_w)[0]; _cq = T(BODY_CAM.data.quat_w_world)[0:1]\n'
  '                        _fw = quat_apply(_cq, torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]\n'
  '                        _bp, _by = base_pose()\n'
  '                        print(f"[BODYCAM] camera at map ({float(_cp[0] - origin[0]):+.2f},{float(_cp[1] - origin[1]):+.2f}) z {float(_cp[2]):.2f}"\n'
  '                              f" looking ({float(_fw[0]):+.2f},{float(_fw[1]):+.2f},{float(_fw[2]):+.2f}) | base ({float(_bp[0] - origin[0]):+.2f},"\n'
  '                              f"{float(_bp[1] - origin[1]):+.2f}) yaw {math.degrees(_by):+.0f}", flush=True)\n'
  '                    except Exception as _e:\n'
  '                        print(f"[BODYCAM] pose debug failed: {_e}", flush=True)\n'),
]
for old, new in R:
    assert s.count(old) == 1, f"[FIX] FAILED: pattern found {s.count(old)} times: {old[:60]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] body camera pose debug + unknown place falls back to bin")
