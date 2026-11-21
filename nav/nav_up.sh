#!/bin/bash
# One terminal for the whole ROS side: relay + map_server (+ AMCL) + Nav2 + goal bridge + RViz. Ctrl+C stops all.
# Usage: nav_up.sh          -> map->odom = identity (ground-truth localisation)
#        nav_up.sh --amcl   -> AMCL localises on the lidar scan; Isaac must send drifting odometry (full_demo_perc_nav_amcl.py)
source /opt/ros/humble/setup.bash
N=@SPOT_DEMO_ROOT@/nav
L=/tmp/nav_logs
mkdir -p $L
rm -f $L/*.log
AMCL=0; [ "$1" == "--amcl" ] && AMCL=1
trap 'echo "[UP] stopping everything"; kill 0' EXIT INT TERM
python3 $N/gen_map.py | grep -E "MAP\]" || exit 1      # map always matches nav_world / scene.json
if [ $AMCL == 1 ]; then
  python3 $N/make_amcl_params.py || exit 1
  P=$N/config/nav2_params_amcl.yaml
  python3 $N/spot_ros_relay.py --amcl > $L/relay.log 2>&1 &
else
  P=$N/config/nav2_params.yaml
  python3 $N/spot_ros_relay.py > $L/relay.log 2>&1 &
fi
sleep 3
ros2 run nav2_map_server map_server --ros-args -p yaml_filename:=$N/maps/nav_room.yaml -p use_sim_time:=true > $L/map.log 2>&1 &
if [ $AMCL == 1 ]; then
  ros2 run nav2_amcl amcl --ros-args --params-file $P > $L/amcl.log 2>&1 &
  ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_localization \
    -p node_names:="[map_server, amcl]" -p autostart:=true -p use_sim_time:=true > $L/map_lm.log 2>&1 &
else
  ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_map \
    -p node_names:="[map_server]" -p autostart:=true -p use_sim_time:=true > $L/map_lm.log 2>&1 &
fi
sleep 3
ros2 launch nav2_bringup navigation_launch.py use_sim_time:=true use_composition:=False autostart:=true \
  params_file:=$P > $L/nav2.log 2>&1 &
python3 $N/goal_bridge.py > $L/bridge.log 2>&1 &
rviz2 -d /opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz --ros-args -p use_sim_time:=true > $L/rviz.log 2>&1 &
echo "[UP] started relay, map_server$([ $AMCL == 1 ] && echo ', AMCL'), Nav2, goal bridge, RViz - params $P - logs in $L  (Ctrl+C stops all)"
tail -F $L/relay.log $L/map_lm.log $L/amcl.log $L/nav2.log $L/bridge.log 2>/dev/null | grep --line-buffered -iE "relay up|bridge up|goal [0-9]+|sim t=|nodes are active|initial pose|error|fail|abort|succeed|exception"
