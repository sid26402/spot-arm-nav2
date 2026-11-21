#!/bin/bash
# Rebuild the whole demo from the perception demo (full_demo_perc.py) in one go. Safe to run any time.
# Output: full_demo_mover.py (+ full_demo_main.py = the same file under a stable name). The original full_demo_perc.py is never edited.
set -e
C=${CODE:-@SPOT_DEMO_ROOT@}
cd "$C"
step() { echo "[BUILD] $*"; }
step "1/10 Nav2 legs           full_demo_perc.py -> full_demo_perc_nav.py";  python3 build/make_perc_nav.py full_demo_perc.py full_demo_perc_nav.py
step "2/10 lidar + crate       nav_link / nav_world";                       python3 build/make_nav_patch.py "$C/nav"
step "3/10 lidar demo          -> full_demo_perc_nav_scan.py";               python3 build/make_perc_nav_scan.py full_demo_perc_nav.py full_demo_perc_nav_scan.py
step "4/10 lidar fixes";                                                      python3 build/fix_scan.py "$C"
step "5/10 odometry drift/AMCL -> full_demo_perc_nav_amcl.py";               python3 build/make_amcl_patch.py "$C"
step "6/10 general go_to       -> full_demo_goto.py";                        python3 build/make_goto_patch.py "$C"
step "7/10 moving obstacle     -> full_demo_mover.py";                       python3 build/make_mover_patch.py "$C"; python3 build/fix_mover_scan.py "$C"; python3 build/fix_mover2.py "$C"
step "8/10 operator panel    -> full_demo_panel.py";                     python3 build/make_panel_patch.py "$C"
step "9/10 multi-cube scene -> full_demo_multi.py";                     python3 build/make_multicube_patch.py "$C"; python3 build/fix_nav_misc.py "$C"
python3 build/make_flow_patch.py "$C"
python3 build/fix_flow_backoff.py "$C"
python3 build/fix_flow_headon.py "$C"
python3 build/fix_flow_turn.py "$C"
python3 build/fix_flow_yawlog.py "$C"
python3 build/fix_flow_turnshift.py "$C"
python3 build/fix_flow_giveup.py "$C"
python3 build/fix_flow_shiftwalk.py "$C"
python3 build/make_place_patch.py "$C"
python3 build/fix_flow_camdetect.py "$C"
python3 build/fix_flow_bodycam.py "$C"
python3 build/fix_flow_bodylook.py "$C"
python3 build/fix_flow_carry.py "$C"
python3 build/fix_flow_noshift.py "$C"
python3 build/fix_flow_approach.py "$C"
python3 build/fix_flow_approach2.py "$C"
python3 build/fix_flow_approach3.py "$C"
python3 build/fix_flow_standoff.py "$C"
python3 build/fix_flow_bodydbg.py "$C"
python3 build/fix_flow_pickzero.py "$C"
python3 build/fix_flow_placedist.py "$C"
step "10/10 checks";
python3 - "$C" << 'PY'
import sys; C = sys.argv[1]
p = f"{C}/nav/gen_map.py"; s = open(p).read()
old = 'print(f"[MAP] PREPICK (Nav2 goal) = {W.PREPICK_POSE}   PICK (fine approach end) = {W.PICK_POSE}")\n'
if old in s:
    open(p, "w").write(s.replace(old, '')); print("[BUILD] gen_map.py: removed the old fixed PREPICK print")
PY
for f in full_demo_multi.py nav/nav_link.py nav/nav_world.py nav/phys_scan.py nav/goto_planner.py nav/spot_ros_relay.py nav/goal_bridge.py nav/loc_err.py nav/check_scene.py nav/gen_map.py; do
  [ -f "$f" ] && python3 -m py_compile "$f"
done
bash -n nav/nav_up.sh
python3 -m py_compile full_demo_flow.py && cp full_demo_flow.py full_demo_main.py
echo "[BUILD] OK -> $C/full_demo_main.py"
