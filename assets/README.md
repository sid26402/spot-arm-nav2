# Assets (not included - recreate them)
The robot and the bin are modified NVIDIA Isaac Sim 6.0 assets. NVIDIA assets may not be redistributed, so this repo ships
the scripts that recreate them from the originals. Run each in Isaac Sim's Script Editor (Window > Script Editor > File > Open).

| File the demo needs | NVIDIA original | Script | Change |
|---|---|---|---|
| `spot_with_arm_v3.usd` | `Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd` | `tools/make_spot_v3.py` | flattened; articulation root moved to `/base`; arm-link colliders (convex hull on sh0, sh1, el0, el1, wr0; convex decomposition on wr1 + finger so the gripper can close on a cube) |
| `props/small_KLT_hollow.usd` | `Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd` | `tools/make_bin_hollow.py` | bin collider: bounding cube -> exact mesh (objects fall in instead of bouncing off a "lid") |

Both scripts write into `~/spot_assets/` (set `SPOT_ASSETS` for `install.sh` if you use another folder). They never overwrite an
existing file (they write `*_recreated.usd` and print a physics diff against it). `tools/usd_physics_diff.py` lists every physics
difference between the NVIDIA originals and your files. The demo scales the bin 1.6x in x/y at load time.
The cube (DexCube) is loaded directly from the Isaac Sim asset server.
