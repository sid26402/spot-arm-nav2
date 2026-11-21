"""Multi-cube user flow: walk to the table -> list the cubes -> ASK which colour -> sidestep to that cube -> pick ->
bin -> come back -> ASK again, until no cubes are left. Answers: panel buttons, typing the colour + Enter, or --order.
STAND-IN: cube colours/positions come from the sim (prim names) until the perception chat provides camera-based
detect_cubes(); the pick itself already uses the hand camera on the SELECTED cube's mask.
Builds full_demo_flow.py from full_demo_multi.py. Without --multi_flow the demo is unchanged. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
src, dst = f"{C}/full_demo_multi.py", f"{C}/full_demo_flow.py"
s = open(src).read()
R = [
 ('parser.add_argument("--slot_gap", type=float, default=0.30, help="MULTI: distance between cube slots (m)")\n',
  'parser.add_argument("--slot_gap", type=float, default=0.30, help="MULTI: distance between cube slots (m)")\n'
  'parser.add_argument("--multi_flow", action="store_true", help="FLOW: ask which cube, pick it, bin it, repeat until none left (implies --distractors)")\n'
  'parser.add_argument("--order", type=str, default="", help="FLOW: automatic answers, e.g. red,dex,blue (otherwise panel buttons / typing)")\n'),
 ('args, _ = parser.parse_known_args()\n',
  'args, _ = parser.parse_known_args()\n'
  'if args.multi_flow:\n'
  '    args.distractors = True\n'),
 ('_pp = gp.plan(CUBE_XY, W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")    # MULTI: stand in front of the chosen cube\n',
  '_pp = gp.plan((W.BENCH[1], W.BENCH[2]) if args.multi_flow else CUBE_XY, W.STANDOFF, pick_headings(W.BENCH), robot_xy=START_XY, label="pick")    # MULTI/FLOW\n'),
 ('    ids = torch.tensor([int(k) for k, v in _id_map(cam.data.info).items() if "/env_0/Object/" in str(v)],\n',
  '    ids = torch.tensor([int(k) for k, v in _id_map(cam.data.info).items() if SEL["prim"] in str(v)],   # FLOW: selected cube\n'),
 ('cam = u.scene["hand_cam"]\n',
  'cam = u.scene["hand_cam"]\n'
  'SEL = {"colour": None, "prim": "/env_0/Object/"}            # FLOW: which cube the hand camera / arm work on\n'),
 ('# ---------------- PANEL: operator panel (nav side) ----------------\n',
  '# ---------------- FLOW: multi-cube user flow ----------------\n'
  'CUBES = {}                                     # colour -> (scene entity, camera prim filter, (x, y) map frame)\n'
  'if args.multi_flow:\n'
  '    CUBES["dex"] = ("object", "/env_0/Object/", CUBE_XY)\n'
  '    for _cn, _xy in DISTRACT.items():\n'
  '        CUBES[_cn] = (f"cube_{_cn}", f"/env_0/Cube_{_cn}/", _xy)\n'
  'REMAIN = list(CUBES)\n'
  'ORDER = [c.strip().lower() for c in args.order.split(",") if c.strip()]\n'
  'VIEW = (prepick.clone(), bench_park.clone(), PICK_YAW)        # stand in front of the table centre to look and ask\n'
  'FLOW = {"asked": False, "reader": False, "placed": []}\n'
  'def _ask_reader():\n'
  '    while True:\n'
  '        line = sys.stdin.readline()\n'
  '        if not line:\n'
  '            return\n'
  '        if line.strip():\n'
  '            PANEL["req"] = ("pick", line.strip().lower())\n'
  'def _choose(colour, ph):\n'
  '    global obj, bench_park, prepick\n'
  '    if ph != "ASK":\n'
  '        _say(f"not asking right now ({ph})"); return\n'
  '    if colour not in REMAIN:\n'
  '        _say(f"no \'{colour}\' cube on the table - choose from {REMAIN}"); return\n'
  '    ent, prim, xy = CUBES[colour]\n'
  '    r = gp.plan(xy, W.STANDOFF, [PICK_YAW], label=f"pick {colour}")\n'
  '    if not r["ok"]:\n'
  '        _say(f"cannot stand in front of {colour}: {r[\'tried\']}"); return\n'
  '    obj = u.scene[ent]\n'
  '    SEL.update(colour=colour, prim=prim)\n'
  '    bench_park = origin + torch.tensor(r["stand"][:2], device=dev)\n'
  '    prepick = origin + torch.tensor(r["pre"][:2], device=dev)\n'
  '    P["tries"] = 0; state["place_tries"] = 0\n'
  '    pp, _ = base_pose()\n'
  '    far = float(torch.norm(bench_park - pp))\n'
  '    _say(f"picking the {colour} cube" + (f" - stepping {far:.2f} m to stand in front of it" if far > 0.05 else ""))\n'
  '    enter("SHIFT" if far > 0.05 else "UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)\n'
  'def _after_approach():\n'
  '    if args.multi_flow and SEL["colour"] is None:\n'
  '        enter("ASK", hold_xy=bench_park, hold_yaw=PICK_YAW)\n'
  '    else:\n'
  '        enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW)\n'
  'def _ask_tick(t):\n'
  '    if t == 1:\n'
  '        seen = ", ".join(f"{c} at ({CUBES[c][2][0]:+.2f},{CUBES[c][2][1]:+.2f})" for c in REMAIN)\n'
  '        print(f"[MULTI] cubes on the table (STAND-IN: from the sim until camera colour detection exists): {seen}", flush=True)\n'
  '        print(f"[ASK] Which cube should I pick? {\' / \'.join(REMAIN)}  -> panel button, type it + Enter, or --order", flush=True)\n'
  '        PANEL["msg"] = f"Which cube? {\' / \'.join(REMAIN)}"\n'
  '        if not FLOW["reader"] and not ORDER:\n'
  '            threading.Thread(target=_ask_reader, daemon=True).start(); FLOW["reader"] = True\n'
  '    if ORDER and t == int(1.0 / dt):\n'
  '        c = ORDER.pop(0)\n'
  '        print(f"[ASK] automatic answer (--order): {c}", flush=True)\n'
  '        _choose(c, "ASK")\n'
  'def _next_round():\n'
  '    global obj, bench_park, prepick\n'
  '    if SEL["colour"] in REMAIN:\n'
  '        REMAIN.remove(SEL["colour"]); FLOW["placed"].append(SEL["colour"])\n'
  '    if not REMAIN:\n'
  '        print(f"[MULTI] all cubes done: {FLOW[\'placed\']}", flush=True); _say("all cubes done"); return\n'
  '    SEL.update(colour=None)\n'
  '    prepick, bench_park = VIEW[0].clone(), VIEW[1].clone()\n'
  '    _say(f"back to the table - left: {REMAIN}")\n'
  '    enter("NAV_TABLE")\n'
  + '# ---------------- PANEL: operator panel (nav side) ----------------\n'),
 ('        elif kind == "cancel":\n',
  '        elif kind == "pick":\n'
  '            _choose(r[1], ph)\n'
  '        elif kind == "cancel":\n'),
 ('            ui.Button("Cancel / stop here", height=28, clicked_fn=lambda: _req("cancel"))\n',
  '            ui.Button("Cancel / stop here", height=28, clicked_fn=lambda: _req("cancel"))\n'
  '            if args.multi_flow:\n'
  '                ui.Label("Which cube to pick (when asked):", height=18)\n'
  '                with ui.HStack(height=30, spacing=4):\n'
  '                    for _c in list(CUBES):\n'
  '                        ui.Button(f"Pick {_c}", clicked_fn=lambda c=_c: _req("pick", c))\n'),
 ('                a, legs = drive(bench_park, PICK_YAW, stow, True, lambda: enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW))\n',
  '                a, legs = drive(bench_park, PICK_YAW, stow, True, _after_approach)\n'),
 ('            elif ph == "IDLE":                                       # PANEL: standing, arm stowed, waiting for a button\n',
  '            elif ph == "ASK":                                        # FLOW: standing at the table, waiting for a colour\n'
  '                a = arm_action(stow, True)\n'
  '                _ask_tick(t)\n'
  '            elif ph == "SHIFT":                                      # FLOW: sidestep to stand in front of the chosen cube\n'
  '                a, legs = drive(bench_park, PICK_YAW, stow, True, lambda: enter("UNFOLD", hold_xy=bench_park, hold_yaw=PICK_YAW), timeout=10.0)\n'
  '            elif ph == "NAV_TABLE":                                  # FLOW: back from the bin to the table view pose\n'
  '                a = arm_action(stow, True)\n'
  '                legs = nav_phase(prepick, PICK_YAW, lambda: enter("APPROACH"), nav_cmd)\n'
  '            elif ph == "IDLE":                                       # PANEL: standing, arm stowed, waiting for a button\n'),
 ('            if ph not in ("NAV_PICK", "NAV_BIN", "NAV_FREE") and not args.no_script_bias:\n',
  '            if ph not in ("NAV_PICK", "NAV_BIN", "NAV_FREE", "NAV_TABLE") and not args.no_script_bias:\n'),
 ('                    for _cn, (_x0, _y0) in DISTRACT.items():           # MULTI: did the pick disturb the neighbours?\n',
  '                    for _cn, (_x0, _y0) in ({} if args.multi_flow else DISTRACT).items():   # MULTI: did the pick disturb the neighbours?\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
# after the RESULT block (end of RETRACT): next round
o = ("                        print(f\"[MULTI] neighbour {_cn}: moved {_mv * 100:.1f} cm, z={float(_q[2]):.3f} -> "
     "{'UNTOUCHED' if _mv < 0.02 and float(_q[2]) > BENCH_H + 0.015 else 'DISTURBED'}\", flush=True)\n")
assert s.count(o) == 1, "[MERGE] FAILED: neighbour print line not found"
s = s.replace(o, o + "                    if args.multi_flow:\n                        _next_round()\n")
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R) + 1} changes -> {dst}")
