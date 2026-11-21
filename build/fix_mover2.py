"""(1) Leave the moving box out of the PhysX ray casts (they keep seeing it at its start pose = ghost); the live-pose box stays.
(2) After STOW, check with the hand camera that the cube is still held; if not, retry LOOK/PICK at the bench.
Patches full_demo_mover.py in place; safe to run twice. Run after fix_mover_scan.py."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_mover.py"
s = open(p).read()
o = "exclude=(_robot_path,),"
if o in s:
    s = s.replace(o, 'exclude=(_robot_path, "/World/envs/env_0/Mover"),', 1)
    print("[FIX] PhysX scan ignores the moving box (no ghost at its start pose)")
else:
    print("[FIX] scan exclude already patched" if '"/World/envs/env_0/Mover"' in s else "[FIX] WARNING: scan exclude line not found")
o = ('                if S["phase"] == "BACKOFF":\n'
     '                    pp, _ = base_pose()\n'
     '                    S["bk"] = prepick.clone()                              # GOTO: back off to the checked pre-pose\n')
n = ('                if S["phase"] == "BACKOFF":                                # PERC guard: cube still in hand after STOW?\n'
     '                    _dp, _mk = _cube_mask()\n'
     '                    _px = int(_mk.sum())\n'
     '                    _dm = float(_dp[_mk].median()) if _px else float("nan")\n'
     '                    if _px >= args.inhand_px // 2 and _dm < args.inhand_d:\n'
     '                        print(f"[PERC] after STOW: cube still in hand ({_px} px at {_dm:.2f} m)", flush=True)\n'
     '                        S["bk"] = prepick.clone()                          # GOTO: back off to the checked pre-pose\n'
     '                    else:\n'
     '                        retry_or_stop(f"cube lost during STOW (hand camera {_px} px, depth {_dm:.2f} m)")\n')
if o in s:
    s = s.replace(o, n, 1)
    print("[FIX] after-STOW hand-camera check added (retries the pick if the cube slipped)")
else:
    print("[FIX] STOW check already added" if "cube lost during STOW" in s else "[FIX] WARNING: STOW block not found")
open(p, "w").write(s)
