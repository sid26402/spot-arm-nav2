"""FLOW fixes after the first body-camera run:
1) BACKOFF while holding the cube: gentle start (1 s ramp) and max 0.2 m/s (was up to 0.4 m/s at once). The red cube was in
   hand after STOW (74934 px) and on the floor 1 s into BACKOFF. Timeout 4 s -> 6 s for the slower walk.
2) A cube that ends MISSED (not in the bin) is reported as failed, not placed.
3) Body camera: the image save no longer runs a second detection (duplicate [DETECT] lines).
Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "CARRY: gentle start" in s:
    print("[FIX] carry/backoff fixes already there"); sys.exit(0)
R = [
 ('                a, legs = drive(S["bk"], PICK_YAW, stow, False, lambda: enter("NAV_BIN"), timeout=4.0)\n',
  '                a, legs = drive(S["bk"], PICK_YAW, stow, False, lambda: enter("NAV_BIN"), timeout=6.0)\n'
  '                _rp = min(1.0, t * dt / 1.0)                      # CARRY: gentle start, max 0.2 m/s while holding the cube\n'
  '                legs = [max(-0.2, min(0.2, legs[0])) * _rp, max(-0.2, min(0.2, legs[1])) * _rp, legs[2]]\n'),
 ('        REMAIN.remove(SEL["colour"]); FLOW["placed"].append(f"{SEL[\'colour\']}->{FLOW.get(\'target\', \'bin\')}")\n',
  '        REMAIN.remove(SEL["colour"])\n'
  '        if globals().get("inside", True):\n'
  '            FLOW["placed"].append(f"{SEL[\'colour\']}->{FLOW.get(\'target\', \'bin\')}")\n'
  '        else:                                          # MISSED: the cube is not in the bin\n'
  '            FLOW.setdefault("failed", []).append(f"{SEL[\'colour\']} (missed)")\n'),
]
o3 = '                    _detect_with(BODY_CAM, image_tag="bodycam")\n'
if o3 in s:
    R.append((o3, '                    _ck = cam; cam = BODY_CAM; save_cam_image("bodycam"); cam = _ck   # image only, detection below\n'))
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] carry: gentle BACKOFF (0.2 m/s, 1 s ramp); MISSED counts as failed; one body-camera detection per look")
