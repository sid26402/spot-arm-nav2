"""Moving obstacle: a kinematic box (not in the Nav2 map) slides into Spot's leg-2 route when Spot gets near.
Builds full_demo_mover.py from full_demo_goto.py. Safe to run twice."""
import json, os, sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
N = f"{C}/nav"

# 1) nav_world: MOVER definition + scene override
p = f"{N}/nav_world.py"
s = open(p).read()
if "MOVER" not in s:
    s += ('\n# Moving obstacle (NOT in the Nav2 map, only the lidar sees it): during `phase`, when Spot comes within `trigger` m of\n'
          '# `target`, the box slides from `start` to `target` at `speed` m/s; it pauses while closer than `keep_away` m to Spot.\n'
          'MOVER = {"size": (0.5, 0.5, 0.6), "start": (2.6, -1.6), "target": (0.75, -1.6), "phase": "NAV_BIN",\n'
          '         "trigger": 2.2, "speed": 0.4, "keep_away": 0.9}\n'
          'if "mover" in SCENE:                       # null = no moving obstacle, or a dict like MOVER\n'
          '    MOVER = None if SCENE["mover"] is None else {**MOVER, **SCENE["mover"]}\n')
    open(p, "w").write(s)
    print("[PATCH] nav_world.py: MOVER added")
else:
    print("[PATCH] nav_world.py already has MOVER")
for name in ("bench_north_east", "bin_east"):          # presets tested without the mover: keep them that way
    f = f"{N}/scenes/{name}.json"
    if os.path.exists(f):
        sc = json.load(open(f)); sc["mover"] = None; json.dump(sc, open(f, "w"), indent=1)
print("[PATCH] scenes bench_north_east / bin_east: mover off")

# 2) demo
src, dst = f"{C}/full_demo_goto.py", f"{C}/full_demo_mover.py"
s = open(src).read()
R = [
 ('parser.add_argument("--no_drift", action="store_true", help="ODOM: send exact odometry (no drift)")\n',
  'parser.add_argument("--no_drift", action="store_true", help="ODOM: send exact odometry (no drift)")\n'
  'parser.add_argument("--no_mover", action="store_true", help="MOVER: no moving obstacle")\n'
  'parser.add_argument("--mover_test", action="store_true", help="MOVER: slide the box once with Spot standing, report, quit")\n'),
 ('    print(f"[SCAN] surprise obstacles (in the sim, NOT in the Nav2 map): {[(n, cx, cy) for n, cx, cy, *_ in W.SURPRISE]}", flush=True)\n',
  '    print(f"[SCAN] surprise obstacles (in the sim, NOT in the Nav2 map): {[(n, cx, cy) for n, cx, cy, *_ in W.SURPRISE]}", flush=True)\n'
  'USE_MOVER = bool(getattr(W, "MOVER", None)) and not args.no_mover\n'
  'if USE_MOVER:                                 # kinematic box: moved by code each step, cannot be pushed, seen by the PhysX lidar\n'
  '    from isaaclab.assets import RigidObjectCfg\n'
  '    _m = W.MOVER\n'
  '    cfg.scene.mover = RigidObjectCfg(\n'
  '        prim_path="{ENV_REGEX_NS}/Mover",\n'
  '        init_state=RigidObjectCfg.InitialStateCfg(pos=(_m["start"][0], _m["start"][1], _m["size"][2] / 2)),\n'
  '        spawn=sim_utils.CuboidCfg(size=tuple(_m["size"]),\n'
  '                                  rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),\n'
  '                                  mass_props=sim_utils.MassPropertiesCfg(mass=20.0),\n'
  '                                  collision_props=sim_utils.CollisionPropertiesCfg(),\n'
  '                                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.10, 0.10))))\n'
  '    print(f"[MOVER] box {_m[\'size\']} from {_m[\'start\']} to {_m[\'target\']} at {_m[\'speed\']} m/s, triggered in {_m[\'phase\']} within {_m[\'trigger\']} m (NOT in the Nav2 map)", flush=True)\n'),
 ('obs, _ = env.reset()\np0, y0 = base_pose()\n',
  'obs, _ = env.reset()\np0, y0 = base_pose()\n'
  'mover = u.scene["mover"] if USE_MOVER else None\n'
  'MV = {"state": "wait", "xy": list(W.MOVER["start"]) if USE_MOVER else None, "paused": False, "write": None}\n'
  'def _mover_write(x, y):\n'
  '    pose = T(mover.data.root_pose_w).clone()                # keep z + orientation as the sim has them (any quat order)\n'
  '    pose[0, 0], pose[0, 1] = float(origin[0]) + x, float(origin[1]) + y\n'
  '    if MV["write"] is None:                                  # find the call form this Isaac Lab version accepts\n'
  '        for name, call in (("index(root_pose=)", lambda p: mover.write_root_pose_to_sim_index(root_pose=p)),\n'
  '                           ("index(p)", lambda p: mover.write_root_pose_to_sim_index(p)),\n'
  '                           ("write_root_pose_to_sim(p)", lambda p: mover.write_root_pose_to_sim(p))):\n'
  '            try:\n'
  '                call(pose); MV["write"] = call\n'
  '                print(f"[MOVER] pose write works via {name}", flush=True); return\n'
  '            except Exception as e:\n'
  '                print(f"[MOVER] pose write {name} failed: {type(e).__name__}: {e}", flush=True)\n'
  '        raise RuntimeError("no working pose write for the mover")\n'
  '    MV["write"](pose)\n'
  'def mover_step(ph, force=False):\n'
  '    if mover is None or MV["state"] == "done":\n'
  '        return\n'
  '    m = W.MOVER\n'
  '    p, _ = base_pose()\n'
  '    rx, ry = float(p[0] - origin[0]), float(p[1] - origin[1])\n'
  '    if MV["state"] == "wait":\n'
  '        if force or (ph == m["phase"] and math.hypot(rx - m["target"][0], ry - m["target"][1]) < m["trigger"]):\n'
  '            MV["state"] = "move"\n'
  '            print(f"[MOVER] sliding into the route (Spot at ({rx:+.2f},{ry:+.2f}))", flush=True)\n'
  '        return\n'
  '    x, y = MV["xy"]\n'
  '    dx, dy = m["target"][0] - x, m["target"][1] - y\n'
  '    d = math.hypot(dx, dy)\n'
  '    near = math.hypot(rx - x, ry - y) < m["keep_away"]\n'
  '    if near != MV["paused"]:\n'
  '        MV["paused"] = near\n'
  '        print(f"[MOVER] {\'paused - Spot too close\' if near else \'moving again\'}", flush=True)\n'
  '    if near:\n'
  '        return\n'
  '    stp = m["speed"] * dt\n'
  '    if d <= stp:\n'
  '        x, y = m["target"]; MV["state"] = "done"\n'
  '        print(f"[MOVER] in place at ({x:+.2f},{y:+.2f}) - blocking the original route", flush=True)\n'
  '    else:\n'
  '        x, y = x + dx / d * stp, y + dy / d * stp\n'
  '    MV["xy"] = [x, y]\n'
  '    _mover_write(x, y)\n'
  'if args.mover_test:\n'
  '    if mover is None:\n'
  '        print("[MOVER] test: no mover in this scene / --no_mover", flush=True)\n'
  '    else:\n'
  '        for k in range(int(6.0 / dt)):\n'
  '            mover_step("TEST", force=(k == 0))\n'
  '            legs_term.cmd[:] = torch.tensor([0.0, 0.0, args.wz_bias], device=dev)\n'
  '            obs, _, _, _, _ = env.step(arm_action(ready, True))\n'
  '            if k % int(1.0 / dt) == 0:\n'
  '                mp = T(mover.data.root_pos_w)[0]\n'
  '                print(f"[MOVER] test t={k * dt:.0f}s commanded ({MV[\'xy\'][0]:+.2f},{MV[\'xy\'][1]:+.2f}) sim ({float(mp[0] - origin[0]):+.2f},{float(mp[1] - origin[1]):+.2f}) z={float(mp[2]):.2f}", flush=True)\n'
  '        if args.scan == "physx":\n'
  '            pr = _scan.hit_paths\n'
  '            _scan.hit_paths = {}\n'
  '            _ = _scan(float(MV["xy"][0]) + 2.0, float(MV["xy"][1]), math.pi)  # look at the box from 2 m east (open side)\n'
  '            print(f"[MOVER] test: lidar from 2 m east sees {sorted(_scan.hit_paths)}", flush=True)\n'
  '    print("[MOVER] test done - quitting", flush=True)\n'
  '    env.close(); app.close(); sys.exit(0)\n'),
 ('            if ph not in ("NAV_PICK", "NAV_BIN") and not args.no_script_bias:\n',
  '            mover_step(ph)\n'
  '            if ph not in ("NAV_PICK", "NAV_BIN") and not args.no_script_bias:\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
