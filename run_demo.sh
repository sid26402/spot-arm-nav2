#!/bin/bash
# One command for the whole demo: Isaac (this terminal, so you can press g) + the ROS 2 stack in the background, in the right order.
# Usage: @SPOT_DEMO_ROOT@/run_demo.sh [--scene NAME] [--amcl] [--no_mover] [--headless] [any other full_demo_main.py flags]
#   --scene NAME : use @SPOT_DEMO_ROOT@/nav/scenes/NAME.json (default scene if omitted)
#   --amcl       : drifting odometry + AMCL localisation (otherwise exact odometry)
# Ctrl+C (or the end of the demo) stops everything, including the ROS stack.
C=@SPOT_DEMO_ROOT@; N=$C/nav
ARM=@SPOT_DEMO_ROOT@/policies/spot_arm_topdown_v9g/exported/policy.pt
LOG=/tmp/demo_run.log; NAVLOG=/tmp/nav_up.log
SCENE=""; AMCL=""; VIZ="--viz kit"; EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    --amcl) AMCL="--amcl"; shift ;;
    --headless) VIZ="--headless"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
cleanup() {
  pkill -f nav_up.sh; pkill -f spot_ros_relay.py; pkill -f goal_bridge.py; pkill -f nav2_; pkill -f map_server
  pkill -f lifecycle_manager; pkill -f nav2_amcl; pkill -f rviz2; pkill -f "tail -F /tmp/nav_logs"
} 2>/dev/null
cleanup; sleep 1
trap 'echo "[RUN] stopping the ROS stack"; cleanup' EXIT
[ -f "$C/full_demo_main.py" ] || { echo "[RUN] $C/full_demo_main.py missing - run @SPOT_DEMO_ROOT@/build_demo.sh first"; exit 1; }
if [ -n "$SCENE" ]; then
  [ -f "$N/scenes/$SCENE.json" ] || { echo "[RUN] no scene $N/scenes/$SCENE.json (have: $(ls $N/scenes | sed 's/.json//' | tr '\n' ' '))"; exit 1; }
  cp "$N/scenes/$SCENE.json" "$N/scene.json"
else
  rm -f "$N/scene.json"
fi
python3 "$N/check_scene.py" | grep -E "approach heading|UNREACHABLE|RESULT"
python3 "$N/check_scene.py" > /dev/null || { echo "[RUN] scene not runnable - stopping"; exit 1; }
DRIFT="--no_drift"; [ -n "$AMCL" ] && DRIFT=""
rm -f "$LOG" "$NAVLOG"
echo "[RUN] scene: ${SCENE:-default} | localisation: $([ -n "$AMCL" ] && echo "AMCL (drifting odometry)" || echo "exact odometry") | log: $LOG | ROS log: $NAVLOG"
(   # start the ROS stack once Isaac is READY (Isaac first, so the sim clock never jumps back for Nav2)
  until grep -qE "\[DEMO\] READY|\[RESULT\]|Traceback" "$LOG" 2>/dev/null; do sleep 2; done
  grep -q "\[DEMO\] READY" "$LOG" || exit 0
  setsid "$N/nav_up.sh" $AMCL > "$NAVLOG" 2>&1 &
  for i in $(seq 1 90); do
    if [ "$(grep -c 'Managed nodes are active' "$NAVLOG" 2>/dev/null)" -ge 2 ] && grep -q "goal bridge up" "$NAVLOG"; then
      echo; echo "[RUN] >>> ROS stack + Nav2 ready - start your recording, click this terminal and press g (or Start in the \"Spot Nav2 panel\" window) <<<"; exit 0
    fi
    sleep 2
  done
  echo "[RUN] WARNING: Nav2 not ready after 3 min - check $NAVLOG"
) &
env -u ROS_DISTRO -u AMENT_PREFIX_PATH -u PYTHONPATH -u LD_LIBRARY_PATH bash -c "cd @ISAACLAB_DIR@ && ./isaaclab.sh -p $C/full_demo_main.py \
  --task Isaac-SpotArm-V7e-Play --arm $ARM $VIZ --fast_kp 4000 --fast_kd 120 --move_time 1.2 $DRIFT ${EXTRA[*]} 2>&1 \
  | tee $LOG | grep --line-buffered -E '\[DEMO\]|\[NAV\]|\[PERC\]|\[GOTO\]|\[MOVER\]|\[PANEL\]|\[ASK\]|\[MULTI\]|\[MULTI\]|\[ODOM\]|\[PLACE\]|\[RESULT\]|Error|Traceback'"
echo "[RUN] finished - summary:"
grep -E "\[MULTI\]|\[NAV\] Nav2 (goal|reached)|\[MOVER\] (sliding|in place)|\[PERC\] (pick verified|after STOW)|\[RESULT\]|\[ASK\]|\[MULTI\] all" "$LOG"
