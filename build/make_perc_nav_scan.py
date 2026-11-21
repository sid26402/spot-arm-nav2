"""Build full_demo_perc_nav_scan.py from full_demo_perc_nav.py: PhysX lidar scan + surprise obstacle + --scan_test."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
R = []
def rep(old, new):
    R.append((old, new))
rep('parser.add_argument("--no_script_bias", action="store_true", help="NAV: do not add wz_bias to the scripted (non-Nav2) leg commands")\n',
    'parser.add_argument("--no_script_bias", action="store_true", help="NAV: do not add wz_bias to the scripted (non-Nav2) leg commands")\n'
    'parser.add_argument("--scan", choices=["physx", "boxes"], default="physx", help="SCAN: physx = ray casts into the real scene, boxes = old nav_world list")\n'
    'parser.add_argument("--scan_rays", type=int, default=360, help="SCAN: rays per 360 deg scan")\n'
    'parser.add_argument("--scan_h", type=float, default=0.12, help="SCAN: height of the scan plane above the floor (m); below the bin top 0.146")\n'
    'parser.add_argument("--no_surprise", action="store_true", help="SCAN: do not spawn the surprise obstacle (not in the Nav2 map)")\n'
    'parser.add_argument("--scan_test", action="store_true", help="SCAN: compare physx scan vs box scan at the start pose, then quit")\n')
rep('from nav_link import StateLink, HeadingHold, GoalClient\n', 'from nav_link import StateLink, HeadingHold, GoalClient, Scan2D\n')
rep('                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)))))\n',
    '                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.55, 0.6)))))\n'
    'if not args.no_surprise:\n'
    '    for _n, _cx, _cy, _sx, _sy, _h, _spawn in W.SURPRISE:      # in the sim, NOT in the Nav2 map -> only the scan sees it\n'
    '        setattr(cfg.scene, _n, AssetBaseCfg(\n'
    '            prim_path="{ENV_REGEX_NS}/" + _n.capitalize(), init_state=AssetBaseCfg.InitialStateCfg(pos=(_cx, _cy, _h / 2)),\n'
    '            spawn=sim_utils.CuboidCfg(size=(_sx, _sy, _h), collision_props=sim_utils.CollisionPropertiesCfg(),\n'
    '                                      visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.45, 0.05)))))\n'
    '    print(f"[SCAN] surprise obstacles (in the sim, NOT in the Nav2 map): {[(n, cx, cy) for n, cx, cy, *_ in W.SURPRISE]}", flush=True)\n')
rep('link = StateLink(W.OBSTACLES, dt)\n',
    '_box_scan = Scan2D(W.OBSTACLES, n_rays=args.scan_rays)\n'
    'if args.scan == "physx":\n'
    '    from phys_scan import PhysxScan\n'
    '    _robot_path = cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")\n'
    '    _scan = PhysxScan(origin_xy=(float(origin[0]), float(origin[1])), exclude=(_robot_path,),\n'
    '                      n_rays=args.scan_rays, height=args.scan_h)\n'
    'else:\n'
    '    _scan = _box_scan\n'
    'print(f"[SCAN] scan source: {args.scan} | {args.scan_rays} rays | plane z={args.scan_h} m", flush=True)\n'
    'link = StateLink(W.OBSTACLES, dt, scan_fn=_scan)\n')
rep('obs, _ = env.reset()\np0, y0 = base_pose()\n',
    'obs, _ = env.reset()\np0, y0 = base_pose()\n'
    'def scan_report(tag):\n'
    '    p, yaw = base_pose()\n'
    '    rx, ry = rel(p)\n'
    '    t0 = time.time(); rp = _scan(rx, ry, yaw); ms = (time.time() - t0) * 1000\n'
    '    rb = _box_scan(rx, ry, yaw)\n'
    '    both = np.isfinite(rp) & np.isfinite(rb)\n'
    '    diff = np.abs(rp[both] - rb[both])\n'
    '    print(f"[SCAN] {tag} at ({rx:+.2f},{ry:+.2f}) yaw {math.degrees(yaw):+.0f}: {ms:.1f} ms/scan | hits {int(np.isfinite(rp).sum())}/{len(rp)} "\n'
    '          f"| min range {float(np.min(rp)):.2f} m | vs box list: median diff {float(np.median(diff)) if diff.size else float(\'nan\'):.3f} m, "\n'
    '          f"{int((diff > 0.10).sum())} rays differ >10 cm", flush=True)\n'
    '    return rp, rb\n'
    'if args.scan_test:\n'
    '    for _ in range(int(1.0 / dt)):\n'
    '        legs_term.cmd[:] = torch.tensor([0.0, 0.0, 0.0], device=dev)\n'
    '        obs, _, _, _, _ = env.step(arm_action(ready, True))\n'
    '    rp, rb = scan_report("TEST")\n'
    '    n = len(rp)\n'
    '    for k in range(8):\n'
    '        sl = slice(k * n // 8, (k + 1) * n // 8)\n'
    '        print(f"[SCAN]   sector {-180 + 45 * k:+4d}..{-135 + 45 * k:+4d} deg: physx min {float(np.min(rp[sl])):5.2f} m | box min {float(np.min(rb[sl])):5.2f} m", flush=True)\n'
    '    if args.scan == "physx":\n'
    '        print(f"[SCAN]   prims hit: {dict(sorted(_scan.hit_paths.items(), key=lambda kv: -kv[1]))}", flush=True)\n'
    '    print("[SCAN] test done - quitting", flush=True)\n'
    '    env.close(); app.close(); sys.exit(0)\n')
rep('            link.send(step, float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_pw[2]), float(_yw[0]),\n'
    '                      float(_lv[0]), float(_lv[1]), float(_av[2]))\n',
    '            link.send(step, float(_pw[0] - origin[0]), float(_pw[1] - origin[1]), float(_pw[2]), float(_yw[0]),\n'
    '                      float(_lv[0]), float(_lv[1]), float(_av[2]))\n'
    '            if args.scan == "physx" and step % int(10.0 / dt) == 0:\n'
    '                scan_report(f"t={step * dt:.0f}s")\n')
rep('import torch\nimport gymnasium as gym\n', 'import torch\nimport numpy as np\nimport gymnasium as gym\n')
for old, new in R:
    n = s.count(old)
    if n != 1:
        sys.exit(f"[MERGE] FAILED: pattern found {n} times (need 1): {old[:80]!r}")
    s = s.replace(old, new)
open(dst, "w").write(s)
print(f"[MERGE] OK: {len(R)} changes -> {dst}")
