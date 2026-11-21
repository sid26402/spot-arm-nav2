"""Build full_demo_perc_nav.py from full_demo_perc.py: Nav2 walking + hand-camera pick. The original is not modified."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
# V8 STOW: lift perception's extra STOW lines (stowed print + hand_check) out; put back under BACKOFF below
import re as _re
_m8 = _re.search(r'(                if S\["phase"\] == "GO_BIN":\n)((?:                    (?!pp, _ = base_pose\(\)).*\n)+?)(                    pp, _ = base_pose\(\)\n                    S\["wp"\] = )', s)
_mid8 = _m8.group(2) if _m8 else ""
if _mid8:
    s = s[:_m8.start(2)] + s[_m8.end(2):]
    print(f"[MERGE] v8 STOW block: {_mid8.count(chr(10))} extra lines moved under BACKOFF", flush=True)
R = []
def rep(old, new):
    R.append((old, new))

rep('"""Full demo: walk to bench -> pick (v5) -> stow -> walk to bin -> place -> retract.\nPress g in THIS terminal to start (start your screen recording first)."""\n',
    '"""Full demo with Nav2 + hand-camera perception:\nFOLD -> NAV_PICK (Nav2 to pre-pick) -> APPROACH (scripted last 0.5 m) -> UNFOLD -> SETTLE -> LOOK -> PICK (camera, verify, retry)\n-> STOW -> BACKOFF (0.6 m) -> NAV_BIN (Nav2 to pre-place) -> GO_BIN (scripted last 0.5 m) -> PLACE -> RELEASE -> RETRACT.\nNeeds @SPOT_DEMO_ROOT@/nav/nav_up.sh running. Room, bench, bin and start come from @SPOT_DEMO_ROOT@/nav/nav_world.py. --no_nav = scripted walk.\nPress g in THIS terminal to start (start your screen recording first)."""\n')
rep('T_START = time.time()\n', 'T_START = time.time()\nsys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")\n')
rep('parser.add_argument("--bin_y", type=float, default=-1.6, help="bin sideways position (m)")\n',
    'parser.add_argument("--bin_x", type=float, default=None, help="bin x (default: nav_world.BIN)")\n'
    'parser.add_argument("--bin_y", type=float, default=None, help="bin y (default: nav_world.BIN)")\n'
    'parser.add_argument("--start_x", type=float, default=None, help="robot start x (default: nav_world.START)")\n'
    'parser.add_argument("--start_y", type=float, default=None, help="robot start y (default: nav_world.START)")\n'
    'parser.add_argument("--no_nav", action="store_true", help="NAV: scripted straight-line walk instead of Nav2")\n'
    'parser.add_argument("--pre_dist", type=float, default=0.5, help="NAV: Nav2 stops this far before the pick/place stand pose (m)")\n'
    'parser.add_argument("--nav_timeout", type=float, default=150.0, help="NAV: sim seconds per Nav2 goal before the scripted fallback")\n'
    'parser.add_argument("--wz_bias", type=float, default=0.10, help="NAV: walking-policy yaw-rate offset added to every leg command")\n'
    'parser.add_argument("--no_script_bias", action="store_true", help="NAV: do not add wz_bias to the scripted (non-Nav2) leg commands")\n')
rep('from isaaclab_tasks.manager_based.manipulation.lift.config.spot_arm.env_cfg import BENCH_X, BENCH_H, CUBE_Z, LIFTED_Z\n',
    'from isaaclab_tasks.manager_based.manipulation.lift.config.spot_arm.env_cfg import BENCH_X, BENCH_H, CUBE_Z, LIFTED_Z\n'
    'import nav_world as W\n'
    'from nav_link import StateLink, HeadingHold, GoalClient\n')
rep('BIN_REL = (args.shift, args.bin_y, BIN_H / 2)\n',
    'BIN_X = W.BIN[1] if args.bin_x is None else args.bin_x\n'
    'BIN_Y = W.BIN[2] if args.bin_y is None else args.bin_y\n'
    'START_XY = (W.START[0] if args.start_x is None else args.start_x, W.START[1] if args.start_y is None else args.start_y)\n'
    'if abs(args.shift - W.SHIFT) > 1e-6 or abs(BIN_X - W.BIN[1]) > 1e-6 or abs(BIN_Y - W.BIN[2]) > 1e-6:\n'
    '    print(f"[DEMO] WARNING: bench/bin ({args.shift},{BIN_X},{BIN_Y}) differ from the Nav2 map ({W.SHIFT},{W.BIN[1]},{W.BIN[2]})", flush=True)\n'
    'BIN_REL = (BIN_X, BIN_Y, BIN_H / 2)\n'
    '_rz = cfg.scene.robot.init_state.pos[2]\n'
    'cfg.scene.robot.init_state.pos = (START_XY[0], START_XY[1], _rz)\n'
    'print(f"[DEMO] robot start ({START_XY[0]:+.2f},{START_XY[1]:+.2f}) z={_rz:.2f}", flush=True)\n'
    'for _n, _cx, _cy, _sx, _sy, _h, _spawn in W.OBSTACLES:          # walls / baffles, same boxes as the Nav2 map\n'
    '    if _spawn:\n'
    '        setattr(cfg.scene, _n, AssetBaseCfg(\n'
    '            prim_path="{ENV_REGEX_NS}/" + _n.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_cx, _cy, _h / 2)),\n'
    '            spawn=sim_utils.CuboidCfg(size=(_sx, _sy, _h), collision_props=sim_utils.CollisionPropertiesCfg(),\n'
    '                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)))))\n')
rep('BIN_TOP, BIN_HALF = BIN_H, (0.10, 0.15)\n',
    '# ---- NAV: link to the ROS 2 side (relay + Nav2 + goal bridge) ----\n'
    'link = StateLink(W.OBSTACLES, dt)\n'
    'hh = HeadingHold(k=3.0, wz_bias=args.wz_bias)\n'
    'goals = GoalClient()\n'
    'prepick = bench_park - torch.tensor([args.pre_dist, 0.0], device=dev)\n'
    'preplace = bin_park + torch.tensor([0.0, args.pre_dist], device=dev)\n'
    'def rel(v):\n'
    '    return round(float(v[0] - origin[0]), 3), round(float(v[1] - origin[1]), 3)\n'
    'print(f"[DEMO] walking: {\'scripted (--no_nav)\' if args.no_nav else \'Nav2\'} | pre-pick {rel(prepick)} yaw 0 | "\n'
    '      f"pick park {rel(bench_park)} | pre-place {rel(preplace)} yaw -90 | bin park {rel(bin_park)}", flush=True)\n'
    '\n'
    'BIN_TOP, BIN_HALF = BIN_H, (0.10, 0.15)\n')
rep('obs, _ = env.reset()\np0, y0 = base_pose()\n',
    'def nav_phase(target_w, yaw_t, on_done, nav_cmd):\n'
    '    """One Nav2 goal; follow its /cmd_vel until SUCCEEDED. On failure or timeout: log and continue with the scripted walk."""\n'
    '    S = state\n'
    '    if args.no_nav:\n'
    '        on_done()\n'
    '        return [0.0, 0.0, 0.0]\n'
    '    if not S.get("goal_sent"):\n'
    '        x, y = rel(target_w)\n'
    '        gid = goals.go(x, y, yaw_t)\n'
    '        S["goal_sent"] = True\n'
    '        print(f"[NAV] Nav2 goal {gid}: ({x:+.2f},{y:+.2f}) yaw {math.degrees(yaw_t):+.0f} deg", flush=True)\n'
    '    st = goals.update()\n'
    '    _, yaw = base_pose()\n'
    '    legs = hh(nav_cmd, yaw)                       # includes wz_bias + heading hold at zero command\n'
    '    S["info"] = f"nav={st} dist_left={goals.dist:.2f} nav_cmd=({nav_cmd[0]:+.2f},{nav_cmd[2]:+.2f})"\n'
    '    if st == "SUCCEEDED":\n'
    '        print("[NAV] Nav2 reached the goal", flush=True)\n'
    '        on_done()\n'
    '    elif st in ("ABORTED", "REJECTED", "CANCELED") or S["t"] * dt > args.nav_timeout:\n'
    '        print(f"[NAV] Nav2 {st} after {S[\'t\'] * dt:.0f}s sim - continuing with the scripted walk", flush=True)\n'
    '        goals.cancel()\n'
    '        on_done()\n'
    '    return legs\n'
    '\n'
    'def _to_go_bin():\n'
    '    enter("GO_BIN")\n'
    '    state["wp"], state["wp_i"] = [(bin_park, -math.pi / 2)], 0\n'
    '\n'
    'obs, _ = env.reset()\np0, y0 = base_pose()\n')
rep('    state.update(phase=p, t=0, good=0, q_from=arm_q().clone(), best=1e9, best_t=0)\n',
    '    state.update(phase=p, t=0, good=0, q_from=arm_q().clone(), best=1e9, best_t=0, goal_sent=False)\n')
rep('        with torch.inference_mode():\n            S, t = state, state["t"]\n',
    '        with torch.inference_mode():\n            nav_cmd = link.recv_cmd(step)\n            S, t = state, state["t"]\n')
rep('                a = arm_move(stow, True, "APPROACH")\n',
    '                a = arm_move(stow, True, "NAV_PICK")\n'
    '            elif ph == "NAV_PICK":\n'
    '                a = arm_action(stow, True)\n'
    '                legs = nav_phase(prepick, 0.0, lambda: enter("APPROACH"), nav_cmd)\n')
rep('                a = arm_move(stow, False, "GO_BIN")\n'
    '                if S["phase"] == "GO_BIN":\n'
    '                    pp, _ = base_pose()\n'
    '                    S["wp"] = [(pp - torch.tensor([0.6, 0.0], device=dev), 0.0), (bin_park, -math.pi / 2)]\n'
    '                    S["wp_i"] = 0\n',
    '                a = arm_move(stow, False, "BACKOFF")\n'
    '                if S["phase"] == "BACKOFF":\n'
    '                    pp, _ = base_pose()\n'
    '                    S["bk"] = pp - torch.tensor([0.6, 0.0], device=dev)     # step back from the bench before Nav2 starts\n'
    '            elif ph == "BACKOFF":\n'
    '                a, legs = drive(S["bk"], 0.0, stow, False, lambda: enter("NAV_BIN"), timeout=6.0)\n'
    '            elif ph == "NAV_BIN":\n'
    '                a = arm_action(stow, False)\n'
    '                legs = nav_phase(preplace, -math.pi / 2, _to_go_bin, nav_cmd)\n')
rep('            legs_term.cmd[:] = torch.tensor(legs, device=dev)\n            obs, _, _, _, _ = env.step(a)\n',
    '            if ph not in ("NAV_PICK", "NAV_BIN") and not args.no_script_bias:\n'
    '                legs = [legs[0], legs[1], legs[2] + args.wz_bias]     # walking policy yaw-rate offset (same fix as Nav2 phases)\n'
    '            legs_term.cmd[:] = torch.tensor(legs, device=dev)\n'
    '            obs, _, _, _, _ = env.step(a)\n'
    '            _pw = T(robot.data.root_pos_w)[0]\n'
    '            _, _, _yw = euler_xyz_from_quat(T(robot.data.root_quat_w))\n'
    '            _lv, _av = T(robot.data.root_lin_vel_b)[0], T(robot.data.root_ang_vel_b)[0]\n'
    '            link.send(step, float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_pw[2]), float(_yw[0]),\n'
    '                      float(_lv[0]), float(_lv[1]), float(_av[2]))\n')
for old, new in R:
    n = s.count(old)
    if n != 1:
        sys.exit(f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}")
    s = s.replace(old, new)
if _mid8:
    _k8 = '                if S["phase"] == "BACKOFF":\n                    pp, _ = base_pose()\n                    S["bk"] = pp - torch'
    if s.count(_k8) != 1:
        sys.exit("[MERGE] FAILED: BACKOFF block for the v8 STOW lines not found")
    _i8 = s.index(_k8) + len('                if S["phase"] == "BACKOFF":\n')
    s = s[:_i8] + _mid8 + s[_i8:]
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
