# Recreate small_KLT_hollow.usd from NVIDIA's Isaac Sim 6.0 KLT bin. Run in Isaac Sim's Script Editor.
# Change (measured): the bin's collider used a bounding cube (acts like a lid - cubes bounce off the top);
# it now uses the exact triangle mesh (approximation "none"), so objects fall inside. The demo scales it 1.6x in x/y at load.
# Output: @SPOT_ASSETS@/props/small_KLT_hollow.usd (or *_recreated.usd if it already exists - never overwritten).
import os
from pxr import Usd, UsdPhysics
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
SRC = f"{get_assets_root_path()}/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd"
DST, existed = out_path(os.path.expanduser("@SPOT_ASSETS@/props"), "small_KLT_hollow.usd")
st = Usd.Stage.Open(Usd.Stage.Open(SRC).Flatten())
top = st.GetDefaultPrim()
prim = st.GetPrimAtPath(top.GetPath().AppendPath("Visuals/FOF_Mesh_Magenta_Box"))
assert prim.IsValid(), "missing Visuals/FOF_Mesh_Magenta_Box in the source bin"
UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")
st.GetRootLayer().Export(DST)
print(f"[ASSET] wrote {DST}")
if existed:
    compare(DST, DST.replace("_recreated.usd", ".usd"), "recreated vs your existing small_KLT_hollow.usd")
