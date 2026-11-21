"""Operator panel (nav side): an omni.ui window in the Isaac viewer with Start, Go to bench / bin / x,y,yaw, Cancel and live status.
Builds full_demo_panel.py from full_demo_mover.py. Keyboard g still works. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
src, dst = f"{C}/full_demo_mover.py", f"{C}/full_demo_panel.py"
s = open(src).read()
R = [
 # 1) panel state, request handling, window (after reset: planner results, enter(), nav_phase() all exist)
 ('obs, _ = env.reset()\np0, y0 = base_pose()\n',
  'obs, _ = env.reset()\np0, y0 = base_pose()\n'
  '# ---------------- PANEL: operator panel (nav side) ----------------\n'
  'PANEL = {"req": None, "msg": "ready: press Start (or g), or a Go button", "pre": None, "stand": None, "yaw": 0.0, "ui": None, "label": ""}\n'
  'FREE_OK = ("WAIT", "IDLE", "DONE")\n'
  'def _req(*r):\n'
  '    PANEL["req"] = r\n'
  'def _say(m):\n'
  '    PANEL["msg"] = m\n'
  '    print(f"[PANEL] {m}", flush=True)\n'
  'def _free_after_nav():\n'
  '    if not args.no_nav and goals.status != "SUCCEEDED":\n'
  '        _say(f"Nav2 {goals.status} - stopped where I am"); _free_done(arrived=False); return\n'
  '    if PANEL["stand"] is not None:\n'
  '        enter("FINE_FREE")\n'
  '    else:\n'
  '        _free_done()\n'
  'def _free_done(arrived=True):\n'
  '    pp, py = base_pose()\n'
  '    enter("IDLE", hold_xy=pp, hold_yaw=py)\n'
  '    if arrived:\n'
  '        _say(f"arrived: {PANEL[\'label\']} at ({float(pp[0] - origin[0]):+.2f},{float(pp[1] - origin[1]):+.2f}) yaw {math.degrees(py):+.0f}")\n'
  'def _go(label, pre_w, stand_w, yaw):\n'
  '    PANEL.update(label=label, pre=pre_w, stand=stand_w, yaw=yaw)\n'
  '    _say(f"going to {label}")\n'
  '    enter("FOLD_FREE")\n'
  'def panel_tick(ph):\n'
  '    r, PANEL["req"] = PANEL["req"], None\n'
  '    if r is not None:\n'
  '        kind = r[0]\n'
  '        if kind == "start":\n'
  '            if ph == "WAIT":\n'
  '                start[0] = True; _say("full demo started")\n'
  '            elif ph == "IDLE":\n'
  '                enter("FOLD"); _say("full demo started")\n'
  '            elif ph == "DONE":\n'
  '                _say("demo already finished - restart the app to run it again")\n'
  '            else:\n'
  '                _say(f"busy ({ph}) - Cancel first")\n'
  '        elif kind == "cancel":\n'
  '            if ph in ("WAIT", "IDLE", "DONE"):\n'
  '                _say("nothing to cancel")\n'
  '            else:\n'
  '                goals.cancel()\n'
  '                pp, py = base_pose()\n'
  '                enter("IDLE", hold_xy=pp, hold_yaw=py)\n'
  '                _say(f"cancelled in {ph} - holding position (full demo cannot resume)")\n'
  '        elif ph not in FREE_OK:\n'
  '            _say(f"busy ({ph}) - Cancel first")\n'
  '        elif kind == "bench":\n'
  '            _go("bench (pick stand pose)", prepick, bench_park, PICK_YAW)\n'
  '        elif kind == "bin":\n'
  '            _go("bin (place stand pose)", preplace, bin_park, PLACE_YAW)\n'
  '        elif kind == "xy":\n'
  '            x, y, yd = r[1], r[2], r[3]\n'
  '            d, who = gp.dist(x, y)\n'
  '            if d < gp.turn_r:\n'
  '                _say(f"({x:+.2f},{y:+.2f}) rejected: {d:.2f} m from {who}, need {gp.turn_r} m to turn")\n'
  '            else:\n'
  '                _go(f"({x:+.2f},{y:+.2f}) yaw {yd:+.0f}", origin + torch.tensor([x, y], device=dev), None, math.radians(yd))\n'
  '    if PANEL["ui"] is not None and step % int(0.25 / dt) == 0:\n'
  '        l_ph, l_nav, l_pose, l_msg = PANEL["ui"]\n'
  '        pp, py = base_pose()\n'
  '        l_ph.text = f"phase: {ph}"\n'
  '        l_nav.text = f"Nav2: {goals.status}" + (f"   distance left {goals.dist:.2f} m" if goals.dist >= 0 else "")\n'
  '        l_pose.text = f"Spot: ({float(pp[0] - origin[0]):+.2f}, {float(pp[1] - origin[1]):+.2f})  yaw {math.degrees(py):+.0f} deg"\n'
  '        l_msg.text = PANEL["msg"]\n'
  'try:\n'
  '    if getattr(args, "headless", False):\n'
  '        raise RuntimeError("headless run")\n'
  '    import omni.ui as ui\n'
  '    _w = ui.Window("Spot Nav2 panel", width=360, height=440)\n'
  '    with _w.frame:\n'
  '        with ui.VStack(spacing=6, height=0):\n'
  '            ui.Label("Spot with arm - Nav2 operator panel", height=22)\n'
  '            _l_ph = ui.Label("phase: -", height=18)\n'
  '            _l_nav = ui.Label("Nav2: -", height=18)\n'
  '            _l_pose = ui.Label("Spot: -", height=18)\n'
  '            _l_msg = ui.Label(PANEL["msg"], height=36, word_wrap=True)\n'
  '            ui.Button("Start full demo  (same as g)", height=32, clicked_fn=lambda: _req("start"))\n'
  '            ui.Button("Go to bench  (pick stand pose)", height=28, clicked_fn=lambda: _req("bench"))\n'
  '            ui.Button("Go to bin  (place stand pose)", height=28, clicked_fn=lambda: _req("bin"))\n'
  '            with ui.HStack(height=24, spacing=4):\n'
  '                ui.Label("x", width=14); _fx = ui.FloatField(width=70)\n'
  '                ui.Label("y", width=14); _fy = ui.FloatField(width=70)\n'
  '                ui.Label("yaw", width=30); _fyaw = ui.FloatField(width=60)\n'
  '            ui.Button("Go to x, y, yaw  (map frame, deg)", height=28,\n'
  '                      clicked_fn=lambda: _req("xy", _fx.model.get_value_as_float(), _fy.model.get_value_as_float(), _fyaw.model.get_value_as_float()))\n'
  '            ui.Button("Cancel / stop here", height=28, clicked_fn=lambda: _req("cancel"))\n'
  '    PANEL["ui"] = (_l_ph, _l_nav, _l_pose, _l_msg)\n'
  '    print("[PANEL] operator panel open: window \\"Spot Nav2 panel\\" in the Isaac viewer", flush=True)\n'
  'except Exception as _e:\n'
  '    print(f"[PANEL] no panel ({_e}) - keyboard g still works", flush=True)\n'),
 # 2) free-navigation phases (before FOLD in the main state machine)
 ('            elif ph == "FOLD":\n',
  '            elif ph == "IDLE":                                       # PANEL: standing, arm stowed, waiting for a button\n'
  '                a = arm_action(stow, True)\n'
  '            elif ph == "FOLD_FREE":\n'
  '                a = arm_move(stow, True, "NAV_FREE")\n'
  '            elif ph == "NAV_FREE":\n'
  '                a = arm_action(stow, True)\n'
  '                legs = nav_phase(PANEL["pre"], PANEL["yaw"], _free_after_nav, nav_cmd)\n'
  '            elif ph == "FINE_FREE":\n'
  '                a, legs = drive(PANEL["stand"], PANEL["yaw"], stow, True, _free_done, timeout=15.0)\n'
  '            elif ph == "FOLD":\n'),
 # 3) run the panel every step; Nav2-driven free phase must not get the scripted yaw bias twice
 ('            if ph not in ("NAV_PICK", "NAV_BIN") and not args.no_script_bias:\n',
  '            panel_tick(ph)\n'
  '            if ph not in ("NAV_PICK", "NAV_BIN", "NAV_FREE") and not args.no_script_bias:\n'),
 # 4) [ODOM] lines only when odometry actually drifts
 ('            if step % int(10.0 / dt) == 0:\n                print(f"[ODOM] t=',
  '            if (not args.no_drift) and step % int(10.0 / dt) == 0:\n                print(f"[ODOM] t='),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}"
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
