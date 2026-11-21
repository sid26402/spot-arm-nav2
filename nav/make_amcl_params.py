#!/usr/bin/env python3
"""Write config/nav2_params_amcl.yaml = nav2_params.yaml + an AMCL block (initial pose from nav_world.START)."""
import math, sys
sys.path.insert(0, "@SPOT_DEMO_ROOT@/nav")
import nav_world as W
N = "@SPOT_DEMO_ROOT@/nav"
base = open(f"{N}/config/nav2_params.yaml").read()
x, y, yaw = W.START[0], W.START[1], math.radians(W.START[2])
amcl = f"""
amcl:
  ros__parameters:
    use_sim_time: true
    global_frame_id: map
    odom_frame_id: odom
    base_frame_id: base_link
    scan_topic: scan
    map_topic: map
    robot_model_type: "nav2_amcl::DifferentialMotionModel"
    alpha1: 0.2
    alpha2: 0.2
    alpha3: 0.2
    alpha4: 0.2
    alpha5: 0.2
    laser_model_type: "likelihood_field"
    laser_likelihood_max_dist: 2.0
    laser_min_range: 0.1
    laser_max_range: 10.0
    max_beams: 90
    sigma_hit: 0.2
    z_hit: 0.5
    z_short: 0.05
    z_max: 0.05
    z_rand: 0.5
    min_particles: 500
    max_particles: 2000
    pf_err: 0.05
    pf_z: 0.99
    resample_interval: 1
    recovery_alpha_fast: 0.0
    recovery_alpha_slow: 0.0
    update_min_d: 0.10
    update_min_a: 0.10
    tf_broadcast: true
    transform_tolerance: 1.0
    save_pose_rate: 0.5
    first_map_only: false
    set_initial_pose: true
    initial_pose:
      x: {x}
      y: {y}
      z: 0.0
      yaw: {yaw}
"""
open(f"{N}/config/nav2_params_amcl.yaml", "w").write(base.rstrip() + "\n" + amcl)
print(f"[AMCL] wrote {N}/config/nav2_params_amcl.yaml (initial pose {x}, {y}, {W.START[2]} deg)")
