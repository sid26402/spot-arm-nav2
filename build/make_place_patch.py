"""PLACE: choose WHERE the cube goes - the bin or more bins from scene.json ("bins": {"bin2": [x, y]}).
- nav/nav_world.py: EXTRA_BINS from scene.json, added to OBSTACLES (Nav2 map, planner, box scan).
- full_demo_flow.py: spawns the extra bins (same KLT), plans a stand + pre-pose for each, and in --multi_flow the answer
  can name the place: "red bin2" (typed), "red:bin2" (--order), or the panel's place buttons. Default: bin.
Run after the flow fixes. Safe to run twice."""
import sys
C = sys.argv[1] if len(sys.argv) > 1 else "@SPOT_DEMO_ROOT@"
# ---- 1) nav_world ----
p = f"{C}/nav/nav_world.py"
s = open(p).read()
if "EXTRA_BINS" in s:
    print("[PATCH] nav_world.py already has EXTRA_BINS")
else:
    o = 'OBSTACLES = [BENCH, BIN, DIVIDER, NB1, NB2, NB3, SB1, SB2] + WALLS\n'
    assert s.count(o) == 1, "[PATCH] FAILED: OBSTACLES line not found in nav_world.py"
    s = s.replace(o, 'EXTRA_BINS = [(str(_n), float(_v[0]), float(_v[1]), 0.317, 0.475, 0.146, False)   # PLACE: more bins (scene.json "bins")\n'
                     '              for _n, _v in SCENE.get("bins", {}).items()]\n'
                     'OBSTACLES = [BENCH, BIN, DIVIDER, NB1, NB2, NB3, SB1, SB2] + WALLS + EXTRA_BINS\n')
    open(p, "w").write(s)
    print("[PATCH] nav_world.py: EXTRA_BINS from scene.json (in the Nav2 map and the planner)")
# ---- 2) demo ----
p = f"{C}/full_demo_flow.py"
s = open(p).read()
if "PLACES = {" in s:
    print("[FIX] place targets already there"); sys.exit(0)
R = [
 ('# ---- PERC: hand camera on the wrist',
  'for _bn, _bx, _by, *_br in getattr(W, "EXTRA_BINS", []):      # PLACE: extra bins, same KLT as the main bin\n'
  '    setattr(cfg.scene, _bn, AssetBaseCfg(\n'
  '        prim_path="{ENV_REGEX_NS}/" + _bn.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_bx, _by, BIN_H / 2)),\n'
  '        spawn=UsdFileCfg(usd_path="@SPOT_ASSETS@/props/small_KLT_hollow.usd", scale=(args.bin_scale, args.bin_scale, 1.0))))\n'
  '# ---- PERC: hand camera on the wrist'),
 ('preplace = origin + torch.tensor(_pl["pre"][:2], device=dev)\n',
  'preplace = origin + torch.tensor(_pl["pre"][:2], device=dev)\n'
  'PLACES = {"bin": (bin_w.clone(), bin_park.clone(), preplace.clone(), PLACE_YAW)}     # PLACE: name -> (centre, stand, pre-pose, yaw)\n'
  'for _bn, _bx, _by, *_br in getattr(W, "EXTRA_BINS", []):\n'
  '    _r2 = gp.plan((_bx, _by), args.place_dist, place_headings(), robot_xy=_pp["stand"][:2], label=_bn)\n'
  '    if _r2["ok"]:\n'
  '        PLACES[_bn] = (origin + torch.tensor([_bx, _by], device=dev), origin + torch.tensor(_r2["stand"][:2], device=dev),\n'
  '                       origin + torch.tensor(_r2["pre"][:2], device=dev), _r2["stand"][2])\n'
  '        print(f"[PLACE] {_r2[\'reason\']}: stand ({_r2[\'stand\'][0]:+.2f},{_r2[\'stand\'][1]:+.2f}) pre-pose ({_r2[\'pre\'][0]:+.2f},{_r2[\'pre\'][1]:+.2f})", flush=True)\n'
  '    else:\n'
  '        print(f"[PLACE] {_bn}: no stand pose ({_r2[\'reason\']}) - not offered", flush=True)\n'
  'print(f"[PLACE] place targets: {list(PLACES)}", flush=True)\n'),
 ('def _choose(colour, ph):\n'
  '    global obj, bench_park, prepick\n',
  'def _choose(colour, ph):\n'
  '    global obj, bench_park, prepick, bin_w, bin_park, preplace, PLACE_YAW\n'
  '    _parts = colour.replace(":", " ").split()                   # PLACE: "red bin2" / "red:bin2"\n'
  '    colour = _parts[0] if _parts else colour\n'
  '    _tgt = _parts[1] if len(_parts) > 1 else FLOW.get("place_next", "bin")\n'
  '    if _tgt not in PLACES:\n'
  '        _say(f"no place called \'{_tgt}\' - choose from {list(PLACES)}"); return\n'),
 ('    obj = u.scene[ent]\n',
  '    obj = u.scene[ent]\n'
  '    bin_w, bin_park, preplace, PLACE_YAW = [v.clone() if torch.is_tensor(v) else v for v in PLACES[_tgt]]\n'
  '    FLOW["target"] = _tgt\n'),
 ('    _say(f"picking the {colour} cube" + ',
  '    _say(f"picking the {colour} cube -> {_tgt}" + '),
 ("        print(f\"[ASK] Which cube should I pick? {' / '.join(REMAIN)}  -> panel button, type it + Enter, or --order\", flush=True)\n",
  "        print(f\"[ASK] Which cube should I pick? {' / '.join(REMAIN)}  -> panel button, type it + Enter, or --order\", flush=True)\n"
  "        if len(PLACES) > 1:\n"
  "            print(f\"[ASK] ... and where to put it: {' / '.join(PLACES)} (type e.g. 'red bin2'; default {FLOW.get('place_next', 'bin')})\", flush=True)\n"),
 ('        REMAIN.remove(SEL["colour"]); FLOW["placed"].append(SEL["colour"])\n',
  '        REMAIN.remove(SEL["colour"]); FLOW["placed"].append(f"{SEL[\'colour\']}->{FLOW.get(\'target\', \'bin\')}")\n'),
 ('                        ui.Button(f"Pick {_c}", clicked_fn=lambda c=_c: _req("pick", c))\n',
  '                        ui.Button(f"Pick {_c}", clicked_fn=lambda c=_c: _req("pick", c))\n'
  '                if len(PLACES) > 1:\n'
  '                    ui.Label("Put it in (choose before Pick):", height=18)\n'
  '                    with ui.HStack(height=30, spacing=4):\n'
  '                        for _b in list(PLACES):\n'
  '                            ui.Button(_b, clicked_fn=lambda b=_b: (FLOW.update(place_next=b), _say(f"next cube goes to {b}")))\n'),
]
for old, new in R:
    n = s.count(old)
    assert n == 1, f"[FIX] FAILED: pattern found {n} times (need 1): {old[:70]!r}"
    s = s.replace(old, new)
open(p, "w").write(s)
print("[FIX] place targets: bin + scene bins; answer 'red bin2' / --order red:bin2 / panel place buttons")
