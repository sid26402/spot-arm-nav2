# Run in Isaac Sim's Script Editor (Window > Script Editor). READ-ONLY: opens USD files as separate stages, never saves.
# Lists every physics difference between an NVIDIA source asset and your modified copy -> <asset folder>/usd_physics_diff.txt
import os
from pxr import Usd, Sdf
H = os.path.expanduser("~")
A = os.path.expanduser("@SPOT_ASSETS@")          # your asset folder
OUT = os.path.join(A, "usd_physics_diff.txt")
try:
    from isaacsim.storage.native import get_assets_root_path
except Exception:
    from isaacsim.core.utils.nucleus import get_assets_root_path
ROOT = get_assets_root_path()
PAIRS = [  # (label, original, modified)
    ("spot: NVIDIA -> flat", f"{ROOT}/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd", f"{A}/spot_with_arm_flat.usd"),
    ("spot: flat -> v3", f"{A}/spot_with_arm_flat.usd", f"{A}/spot_with_arm_v3.usd"),
    ("spot: NVIDIA -> v3", f"{ROOT}/Isaac/Robots/BostonDynamics/spot/spot_with_arm.usd", f"{A}/spot_with_arm_v3.usd"),
    ("bin: NVIDIA -> hollow", f"{ROOT}/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd", f"{A}/props/small_KLT_hollow.usd"),
]
KEYS = ("physics", "physx", "drive", "limit", "collision", "mass", "density", "material", "approximation", "kinematic")

def snap(path):
    st = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
    if st is None:
        return None, {}
    dp = st.GetDefaultPrim()
    base = str(dp.GetPath()) if dp else ""
    out = {}
    for p in st.Traverse():
        rel = str(p.GetPath())[len(base):] or "/"
        d = {"@schemas": ",".join(sorted(p.GetAppliedSchemas())), "@type": p.GetTypeName()}
        for a in p.GetAttributes():
            n = a.GetName()
            if any(k in n.lower() for k in KEYS) and a.HasAuthoredValue():
                v = a.Get()
                d[n] = str(v)[:80]
        for r in p.GetRelationships():
            n = r.GetName()
            if any(k in n.lower() for k in KEYS):
                d[n] = ",".join(str(t)[len(base):] for t in r.GetTargets())
        out[rel] = d
    return st, out

lines = []
for label, a, b in PAIRS:
    lines.append(f"===== {label}\n  A {a}\n  B {b}")
    sa, A = snap(a)
    sb, B = snap(b)
    if sa is None or sb is None:
        lines.append(f"  could not open {'A' if sa is None else 'B'} - skipped"); continue
    onlyA = sorted(set(A) - set(B)); onlyB = sorted(set(B) - set(A))
    lines.append(f"  prims: A {len(A)}  B {len(B)}  only in A {len(onlyA)}  only in B {len(onlyB)}")
    for p in onlyA[:30]: lines.append(f"  - only in A: {p}  {A[p]['@type']}")
    for p in onlyB[:30]: lines.append(f"  + only in B: {p}  {B[p]['@type']}  {B[p]['@schemas']}")
    n = 0
    for p in sorted(set(A) & set(B)):
        for k in sorted(set(A[p]) | set(B[p])):
            va, vb = A[p].get(k, "<unset>"), B[p].get(k, "<unset>")
            if va != vb:
                lines.append(f"  * {p} :: {k}: {va}  ->  {vb}"); n += 1
                if n > 400: break
        if n > 400:
            lines.append("  ... (truncated at 400 changes)"); break
    lines.append(f"  changed values: {n}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print(f"[USD-DIFF] wrote {OUT} ({len(lines)} lines)")
