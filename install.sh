#!/bin/bash
# Configure this checkout for your machine: fills the @TOKENS@ with real paths and links the custom Isaac Lab code.
# Usage: ISAACLAB_DIR=~/IsaacLab SPOT_ASSETS=/path/to/usd/folder ./install.sh     (run once; re-run after git pull)
set -e
ROOT=$(cd "$(dirname "$0")" && pwd)
ISAACLAB_DIR=${ISAACLAB_DIR:-$HOME/IsaacLab}
SPOT_ASSETS=${SPOT_ASSETS:-$ROOT/assets}
[ -x "$ISAACLAB_DIR/isaaclab.sh" ] || { echo "[INSTALL] Isaac Lab not found at $ISAACLAB_DIR (set ISAACLAB_DIR)"; exit 1; }
for f in spot_with_arm_v3.usd props/small_KLT_hollow.usd; do
  [ -f "$SPOT_ASSETS/$f" ] || echo "[INSTALL] WARNING: $SPOT_ASSETS/$f missing - create it with assets/tools (see assets/README.md)"
done
grep -rlI --exclude=install.sh '@SPOT_DEMO_ROOT@\|@SPOT_ASSETS@\|@ISAACLAB_DIR@' "$ROOT" | grep -v '/\.git/' | while read -r f; do
  sed -i "s#@SPOT_DEMO_ROOT@#$ROOT#g; s#@SPOT_ASSETS@#$SPOT_ASSETS#g; s#@ISAACLAB_DIR@#$ISAACLAB_DIR#g" "$f"
done
link() {  # $1 repo path, $2 Isaac Lab path
  local dst="$ISAACLAB_DIR/$2"
  if [ -e "$dst" ] && [ ! -L "$dst" ]; then echo "[INSTALL] $dst exists and is not a link - left alone"; return; fi
  mkdir -p "$(dirname "$dst")"; ln -sfn "$ROOT/$1" "$dst"; echo "[INSTALL] linked $dst"
}
link isaaclab_ext/locomotion_spot_arm source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot_arm
link isaaclab_ext/lift_spot_arm source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/spot_arm
link isaaclab_ext/custom_velocity_play.py scripts/reinforcement_learning/rsl_rl/custom_velocity_play.py
mkdir -p "$ROOT/logs" "$ROOT/nav/maps"
touch "$ROOT/.configured"
echo "[INSTALL] done. Next: ./run_demo.sh --scene pick_test --multi_flow --body_cam   (see README)"
