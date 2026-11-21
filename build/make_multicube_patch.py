"""Multi-cube test scene: the task cube can sit in one of 3 slots along the bench (spacing 0.30 m), with coloured
neighbour cubes in the other slots (--distractors). Spot's stand pose is planned for the chosen cube, not the bench centre.
Builds full_demo_multi.py from full_demo_panel.py. Defaults (no flags) = unchanged demo. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
src, dst = f"{C}/full_demo_panel.py", f"{C}/full_demo_multi.py"
s = open(src).read()
R = [
 ('parser.add_argument("--no_mover", action="store_true", help="MOVER: no moving obstacle")\n',
  'parser.add_argument("--no_mover", action="store_true", help="MOVER: no moving obstacle")\n'
  'parser.add_argument("--cube_slot", type=int, default=0, choices=[-1, 0, 1], help="MULTI: task cube slot along the bench (-1, 0, +1; spacing 0.30 m)")\n'
  'parser.add_argument("--distractors", action="store_true", help="MULTI: coloured neighbour cubes in the other slots")\n'
  'parser.add_argument("--slot_gap", type=float, default=0.30, help="MULTI: distance between cube slots (m)")\n'),
 ('cfg.scene.table.init_state.pos = (W.BENCH[1], W.BENCH[2], BENCH_H / 2)          # GOTO: bench from nav_world / scene.json\n'
  'cfg.scene.object.init_state.pos = [W.BENCH[1], W.BENCH[2], CUBE_Z]\n',
  '_bax = (1.0, 0.0) if W.BENCH[3] > W.BENCH[4] else (0.0, 1.0)            # MULTI: bench long axis\n'
  'def _slot_xy(k):\n'
  '    return (W.BENCH[1] + _bax[0] * args.slot_gap * k, W.BENCH[2] + _bax[1] * args.slot_gap * k)\n'
  'CUBE_XY = _slot_xy(args.cube_slot)\n'
  'cfg.scene.table.init_state.pos = (W.BENCH[1], W.BENCH[2], BENCH_H / 2)          # GOTO: bench from nav_world / scene.json\n'
  'cfg.scene.object.init_state.pos = [CUBE_XY[0], CUBE_XY[1], CUBE_Z]\n'
  'print(f"[MULTI] task cube in slot {args.cube_slot:+d} at ({CUBE_XY[0]:+.2f},{CUBE_XY[1]:+.2f}) | task cube spawn: {type(cfg.scene.object.spawn).__name__} "\n'
  '      f"{getattr(cfg.scene.object.spawn, \'usd_path\', \'\')} scale {getattr(cfg.scene.object.spawn, \'scale\', None)}", flush=True)\n'
  'DISTRACT = {}\n'
  'if args.distractors:                          # coloured neighbour cubes (dynamic: they can be knocked over -> reported at the end)\n'
  '    from isaaclab.assets import RigidObjectCfg\n'
  '    _cols = {"red": (0.85, 0.08, 0.08), "blue": (0.10, 0.25, 0.90), "green": (0.10, 0.70, 0.20)}\n'
  '    _free = [k for k in (-1, 0, 1) if k != args.cube_slot]\n'
  '    for _k, (_cn, _rgb) in zip(_free, list(_cols.items())):\n'
  '        _x, _y = _slot_xy(_k)\n'
  '        setattr(cfg.scene, f"cube_{_cn}", RigidObjectCfg(\n'
  '            prim_path="{ENV_REGEX_NS}/Cube_" + _cn,\n'
  '            init_state=RigidObjectCfg.InitialStateCfg(pos=(_x, _y, CUBE_Z)),\n'
  '            spawn=sim_utils.CuboidCfg(size=(0.048, 0.048, 0.048),\n'
  '                                      rigid_props=sim_utils.RigidBodyPropertiesCfg(),\n'
  '                                      mass_props=sim_utils.MassPropertiesCfg(mass=0.10),\n'
  '                                      collision_props=sim_utils.CollisionPropertiesCfg(),\n'
  '                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_rgb))))\n'
  '        DISTRACT[_cn] = (_x, _y)\n'
  '    print(f"[MULTI] neighbour cubes: {DISTRACT}", flush=True)\n'),
 ('"pose_range": ({"x": (-0.10, 0.10), "y": (-0.03, 0.03), "z": (0.0, 0.0)} if W.BENCH[3] > W.BENCH[4] else\n'
  '                                                         {"x": (-0.03, 0.03), "y": (-0.10, 0.10), "z": (0.0, 0.0)})}\n',
  '"pose_range": ({"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)} if args.distractors else   # MULTI: no big jitter next to neighbours\n'
  '                                                         {"x": (-0.10, 0.10), "y": (-0.03, 0.03), "z": (0.0, 0.0)} if W.BENCH[3] > W.BENCH[4] else\n'
  '                                                         {"x": (-0.03, 0.03), "y": (-0.10, 0.10), "z": (0.0, 0.0)})}\n'),
 ('_pp = gp.plan((W.BENCH[1], W.BENCH[2]), W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")\n',
  '_pp = gp.plan(CUBE_XY, W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")    # MULTI: stand in front of the chosen cube\n'),
 ('                          f"dy={float(d[1]):+.3f} z={float(cube[2]):.3f} m", flush=True)\n',
  '                          f"dy={float(d[1]):+.3f} z={float(cube[2]):.3f} m", flush=True)\n'
  '                    for _cn, (_x0, _y0) in DISTRACT.items():           # MULTI: did the pick disturb the neighbours?\n'
  '                        _q = T(u.scene[f"cube_{_cn}"].data.root_pos_w)[0]\n'
  '                        _mv = math.hypot(float(_q[0] - origin[0]) - _x0, float(_q[1] - origin[1]) - _y0)\n'
  '                        print(f"[MULTI] neighbour {_cn}: moved {_mv * 100:.1f} cm, z={float(_q[2]):.3f} -> {\'UNTOUCHED\' if _mv < 0.02 and float(_q[2]) > BENCH_H + 0.015 else \'DISTURBED\'}", flush=True)\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
