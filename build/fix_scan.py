"""Fixes found while testing the PhysX lidar (applied by hand on 2026-11-12, now scripted). Safe to run twice.
  - Isaac Lab 3.0: turn on PhysX scene queries (ray casts) - the setting lives on cfg.sim / cfg.sim.physics
  - exclude Spot's own colliders: robot prim path is the regex form env_.* -> env_0
  - --scan_test: report raw PhysX hits; phys_scan.py counts them"""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
p = f"{C}/full_demo_perc_nav_scan.py"
s = open(p).read()
done = []
if "enable_scene_query_support" not in s:
    o = "cfg.episode_length_s = 1.0e6\n"
    assert s.count(o) == 1, "episode_length_s line not found"
    s = s.replace(o, o +
        'if args.scan == "physx":                       # PhysX ray casts need scene queries; Isaac Lab disables them by default\n'
        '    _sq = []\n'
        '    if hasattr(cfg.sim, "enable_scene_query_support"):\n'
        '        cfg.sim.enable_scene_query_support = True; _sq.append("sim")\n'
        '    for _k in dir(cfg.sim):                      # Isaac Lab 3.0: physics settings moved (no cfg.sim.physx)\n'
        '        if _k.startswith("_"):\n'
        '            continue\n'
        '        _v = getattr(cfg.sim, _k, None)\n'
        '        if _v is not None and not isinstance(_v, (int, float, str, bool, tuple, list)) and hasattr(_v, "enable_scene_query_support"):\n'
        '            setattr(_v, "enable_scene_query_support", True); _sq.append(f"sim.{_k}")\n'
        '    if _sq:\n'
        '        print(f"[SCAN] PhysX scene query support: ON ({_sq})", flush=True)\n'
        '    else:\n'
        '        print(f"[SCAN] WARNING: no enable_scene_query_support found. cfg.sim fields: {[k for k in dir(cfg.sim) if not k.startswith(\'_\')]}", flush=True)\n', 1)
    done.append("scene queries")
if "_re.sub" not in s:
    o = '    _robot_path = cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")\n'
    assert s.count(o) == 1, "robot path line not found"
    s = s.replace(o, '    import re as _re\n' + o +
        '    _robot_path = _re.sub(r"env_[^/]*", "env_0", _robot_path)          # regex form env_.* -> env_0\n'
        '    print(f"[SCAN] excluding Spot\'s own colliders under {_robot_path} (cfg prim_path {cfg.scene.robot.prim_path})", flush=True)\n', 1)
    done.append("robot exclude path")
if "raw PhysX hits" not in s:
    o = "        print(f\"[SCAN]   prims hit: {dict(sorted(_scan.hit_paths.items(), key=lambda kv: -kv[1]))}\", flush=True)\n"
    assert s.count(o) == 1, "prims hit line not found"
    s = s.replace(o, o + "        print(f\"[SCAN]   raw PhysX hits {_scan.raw_hits} (on Spot itself {_scan.robot_hits})\", flush=True)\n", 1)
    done.append("raw hit report")
open(p, "w").write(s)
q = f"{C}/nav/phys_scan.py"
t = open(q).read()
if "raw_hits" not in t:
    for o, n in [("        self.hit_paths = {}                       # prim path -> hit count (for the --scan_test report)\n",
                  "        self.hit_paths = {}                       # prim path -> hit count (for the --scan_test report)\n"
                  "        self.raw_hits, self.robot_hits = 0, 0     # every hit PhysX reports / hits on Spot itself (diagnostics)\n"),
                 ("        p = str(getattr(hit, \"collision\", \"\"))\n        if p.startswith(self.excl):\n            return True",
                  "        self.raw_hits += 1\n        p = str(getattr(hit, \"collision\", \"\"))\n        if p.startswith(self.excl):\n            self.robot_hits += 1\n            return True")]:
        assert t.count(o) == 1, "phys_scan pattern not found"
        t = t.replace(o, n)
    open(q, "w").write(t)
    done.append("phys_scan counters")
print(f"[FIX] fix_scan: {', '.join(done) if done else 'already applied'}")
