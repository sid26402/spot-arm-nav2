# Recreate spot_with_arm_v3.usd from NVIDIA's Isaac Sim 6.0 Spot-with-arm model. Run in Isaac Sim's Script Editor.
# Changes (measured with usd_physics_diff.py against the file the policies were trained with):
#  1) articulation root moved from the robot's top prim to its body link /base
#  2) collision added to the arm links: convex hull on sh0, sh1 (2 meshes), el0, el1, wr0;
#     convex DECOMPOSITION on wr1 (gripper jaw) and fngr (finger) so the gripper can actually close around a cube
# Output: @SPOT_ASSETS@/spot_with_arm_v3.usd (or *_recreated.usd if that file already exists - never overwritten).
import os
from pxr import Usd, UsdPhysics, PhysxSchema
H = os.path.expanduser("~")
# ---- shared helpers ----
import os
from pxr import Usd
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from isaacsim.core.utils.nucleus import get_assets_root_path
KEYS = ("physics", "physx", "drive", "limit", "collision", "mass", "density", "material", "approximation", "kinematic")

def snap(path):
    st = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
    if st is None:
        return None
    base = str(st.GetDefaultPrim().GetPath())
    out = {}
    for p in st.Traverse():
        sp = str(p.GetPath())
        if not sp.startswith(base):                    # only what the robot/prop reference brings in (default prim)
            continue
        d = {"@schemas": ",".join(sorted(p.GetAppliedSchemas()))}
        for a in p.GetAttributes():
            n = a.GetName()
            if any(k in n.lower() for k in KEYS) and a.HasAuthoredValue() and not n.startswith("state:"):
                d[n] = str(a.Get())[:80]
        out[sp[len(base):] or "/"] = d
    return out

def compare(a, b, label):
    A, B = snap(a), snap(b)
    if A is None or B is None:
        print(f"[ASSET] {label}: cannot compare (missing file)"); return None
    diffs = [f"only in one: {p}" for p in sorted(set(A) ^ set(B))]
    for p in sorted(set(A) & set(B)):
        for k in sorted(set(A[p]) | set(B[p])):
            if A[p].get(k) != B[p].get(k):
                diffs.append(f"{p} :: {k}: {A[p].get(k)} vs {B[p].get(k)}")
    print(f"[ASSET] {label}: {len(diffs)} physics differences" + ("" if diffs else " - identical"))
    for d in diffs[:20]:
        print("   ", d)
    return len(diffs)

def out_path(folder, name):
    """Never overwrite an existing asset: write <name>_recreated.usd next to it instead."""
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    return (p, False) if not os.path.exists(p) else (p.replace(".usd", "_recreated.usd"), True)
# ---- recipe ----
SRC = f"{get_assets_root_path()}/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd"
DST, existed = out_path(os.path.expanduser("@SPOT_ASSETS@"), "spot_with_arm_v3.usd")
COLLIDERS = {"arm0_link_sh0/visuals": "convexHull", "arm0_link_sh1/visuals/mesh_0": "convexHull",
             "arm0_link_sh1/visuals/mesh_1": "convexHull", "arm0_link_el0/visuals": "convexHull",
             "arm0_link_el1/visuals": "convexHull", "arm0_link_wr0/visuals": "convexHull",
             "arm0_link_wr1/visuals": "convexDecomposition", "arm0_link_fngr/visuals": "convexDecomposition"}
st = Usd.Stage.Open(Usd.Stage.Open(SRC).Flatten())                # one self-contained layer (what flatten_spot.py did)
top = st.GetDefaultPrim()
body = st.GetPrimAtPath(top.GetPath().AppendPath("base"))
assert body.IsValid(), "no /base link in the source model"
top.RemoveAPI(UsdPhysics.ArticulationRootAPI); top.RemoveAPI(PhysxSchema.PhysxArticulationAPI)
UsdPhysics.ArticulationRootAPI.Apply(body); PhysxSchema.PhysxArticulationAPI.Apply(body)
for rel, approx in COLLIDERS.items():
    prim = st.GetPrimAtPath(top.GetPath().AppendPath(rel))
    assert prim.IsValid(), f"missing {rel}"
    c = UsdPhysics.CollisionAPI.Apply(prim); c.CreateCollisionEnabledAttr().Set(True); c.CreateSimulationOwnerRel()
    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(approx)
st.GetRootLayer().Export(DST)
print(f"[ASSET] wrote {DST}")
if existed:
    compare(DST, DST.replace("_recreated.usd", ".usd"), "recreated vs your existing spot_with_arm_v3.usd")
