#!/usr/bin/env python
"""P2: panel-QBX on REAL meshes + zero-shot analytic->mesh transfer of the learned correction.

Leg 1 (known kernel): interior Laplace Dirichlet on classic watertight graphics meshes
(normalized to max radius 1; manufactured harmonic GT from exterior charges works on ANY
watertight mesh). Interior probes = vertices pushed inward, ACCEPTED only if generalized
winding number ~1 (inside) and not too close to the surface.
Leg 2 (headline): Yukawa-given-Laplace correction TRAINED ONLY on analytic charts (spectral
torus+sphere cases from m1_combined) applied ZERO-SHOT on the mesh panel operator — real-mesh,
real-triangulation transfer of the learned kernel object. Controls: wrong-kernel floor.
"""
from __future__ import annotations
import argparse, gc, hashlib, json, math, os
import numpy as np
import torch

import panel_qbx as pq
import m1_combined as m1
import sphere_qbx as sq
import surface_varkernel as vk

PI = math.pi


def _mkdir_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    return {
        "p2_mesh.py": _sha256(__file__),
        "m1_combined.py": _sha256(m1.__file__),
        "panel_qbx.py": _sha256(pq.__file__),
        "sphere_qbx.py": _sha256(sq.__file__),
        "surface_varkernel.py": _sha256(vk.__file__),
    }


def train_correction(kappa, steps, seed, dev):
    """Train the frozen analytic-chart correction used by the mesh fleet."""
    print("[train] correction on analytic torus+sphere (spectral bases)...", flush=True)
    tr = ([m1.make_case("torus", 100 + i, kappa, dev) for i in range(6)]
          + [m1.make_case("sphere", 150 + i, kappa, dev) for i in range(6)])
    torch.manual_seed(seed)
    corr = vk.Corr(h=64, din=11).to(dev)
    opt = torch.optim.Adam(corr.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5)
    rng = np.random.default_rng(seed)
    for s in range(steps):
        c = tr[rng.integers(0, len(tr))]
        loss = vk.rel(vk.solve_eval(c, corr), c["uq"])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(corr.parameters(), 1.0)
        opt.step()
        sched.step()
    corr.eval()
    return corr


def save_correction(path, corr, kappa, steps, seed):
    _mkdir_parent(path)
    state = {name: value.detach().cpu() for name, value in corr.state_dict().items()}
    torch.save(
        {
            "format_version": 1,
            "state_dict": state,
            "seed": int(seed),
            "kappa": float(kappa),
            "steps": int(steps),
            "ntrain_torus": 6,
            "ntrain_sphere": 6,
            "model": {"class": "Corr", "h": 64, "din": 11},
            "source_hashes": source_hashes(),
        },
        path,
    )
    print(f"[checkpoint] saved {path}", flush=True)


def load_correction(path, kappa, dev):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported checkpoint format in {path}")
    if not math.isclose(float(payload["kappa"]), float(kappa), rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"checkpoint kappa={payload['kappa']} does not match requested kappa={kappa}"
        )
    expected_model = {"class": "Corr", "h": 64, "din": 11}
    if payload.get("model") != expected_model:
        raise ValueError(f"checkpoint model metadata mismatch: {payload.get('model')}")
    corr = vk.Corr(h=64, din=11).to(dev)
    corr.load_state_dict(payload["state_dict"], strict=True)
    corr.eval()
    print(
        f"[checkpoint] loaded {path} seed={payload['seed']} steps={payload['steps']}",
        flush=True,
    )
    return corr, payload


def load_obj(path):
    V, F = [], []
    for ln in open(path):
        if ln.startswith("v "):
            V.append([float(x) for x in ln.split()[1:4]])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln.split()[1:]]
            for k in range(1, len(idx) - 1):          # fan-triangulate
                F.append([idx[0], idx[k], idx[k + 1]])
    V = np.array(V); F = np.array(F)
    V -= V.mean(0); V /= np.linalg.norm(V, axis=1).max()
    return V, F


def check_mesh(V, F):
    E = set()
    cnt = {}
    for f in F:
        for a, b in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            e = (min(a, b), max(a, b)); E.add(e); cnt[e] = cnt.get(e, 0) + 1
    watertight = all(c == 2 for c in cnt.values())
    chi = len(V) - len(E) + len(F)
    return watertight, (2 - chi) // 2


def winding(Xq, V, F):
    P0 = V[F]
    e1 = P0[:, 1] - P0[:, 0]; e2 = P0[:, 2] - P0[:, 0]
    cr = np.cross(e1, e2); fa = 0.5 * np.linalg.norm(cr, axis=1)
    fn = cr / (2 * fa[:, None])
    w = np.zeros(len(Xq))
    for g in range(3):
        y = (pq.GAUSS3[g][None, :, None] * P0).sum(1)
        d = y[None, :, :] - Xq[:, None, :]
        r = np.linalg.norm(d, axis=2).clip(1e-15)
        w += ((d * fn[None]).sum(2) / r ** 3 * (fa / 3)[None]).sum(1)
    return w / (4 * PI)


def interior_probes(V, F, vn, h, n_want=150, depth_mult=5.0):
    tau = pq.local_thickness(V, vn, h)
    step = max(1, len(V) // (3 * n_want))
    idx = np.arange(0, len(V), step)
    d = np.minimum(depth_mult * h, 0.35 * tau[idx])          # stay well inside thin parts
    cand = V[idx] - d[:, None] * vn[idx]
    w = winding(cand, V, F)
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(V).query(cand)
    ok = (np.abs(w - 1.0) < 0.05) & (dist > 0.8 * d) & (d > 2 * h)   # drop probes that had to go too shallow
    return cand[ok][:n_want]


def mesh_case(V, F, kappa, seed, dev):
    """Build a solve/eval case dict on the PANEL operator (same schema as m1.make_case)."""
    rng = np.random.default_rng(seed)
    S, vn, fa = pq.assemble_panel_qbx(V, F, sub_levels=3, p=10)
    h = math.sqrt(fa.mean())
    tau = pq.local_thickness(V, vn, h) / h                    # thickness in units of h
    tau_h = dict(min=float(tau.min()), p5=float(np.percentile(tau, 5)),
                 med=float(np.median(tau)))
    Xq = interior_probes(V, F, vn, h)
    c = pq._fib(8) * rng.uniform(2.5, 3.5, (8, 1)); q = rng.normal(size=8); q -= q.mean()
    g = vk.yukawa_gt(V, c, q, kappa); uq = vk.yukawa_gt(Xq, c, q, kappa)
    g0 = pq.u_harm(V, c, q); uq0 = pq.u_harm(Xq, c, q)          # known-kernel (Laplace) variant
    # eval matrix P: 3pt Gauss direct with hat density
    P0 = V[F]
    Pmat = np.zeros((len(Xq), len(V)))
    for gq in range(3):
        y = (pq.GAUSS3[gq][None, :, None] * P0).sum(1)
        D = np.linalg.norm(Xq[:, None, :] - y[None, :, :], axis=2).clip(1e-15)
        Gk = 1.0 / (4 * PI * D) * (fa / 3)[None, :]
        for kv in range(3):
            np.add.at(Pmat.T, F[:, kv], (Gk * pq.GAUSS3[gq, kv]).T)
    wc = np.zeros(len(V))
    for kv in range(3):
        np.add.at(wc, F[:, kv], fa / 3)
    T = lambda a: torch.tensor(a, device=dev)
    Vt, vnt, Xqt = T(V), T(vn), T(Xq)
    return dict(Sbase=T(S), Pbase=T(Pmat), g=T(g), uq=T(uq), g0=T(g0), uq0=T(uq0),
                wc=T(wc), Xc=Vt, nrmc=vnt, Xq=Xqt, tau_h=tau_h)


def solve_eval_mesh(case, corr, chunk=384):
    """Row-chunked correction application (meshes are inference-only; avoids OOM at N~6k)."""
    S = case["Sbase"].clone(); P = case["Pbase"].clone()
    if corr is not None:
        Xc, nrmc, wc, Xq = case["Xc"], case["nrmc"], case["wc"], case["Xq"]
        for i0 in range(0, Xc.shape[0], chunk):
            f = vk.feats(Xc[i0:i0 + chunk], Xc, nrmc[i0:i0 + chunk], nrmc, v2=True)
            S[i0:i0 + chunk] += corr(f) * wc[None, :]
        for i0 in range(0, Xq.shape[0], chunk):
            f = vk.feats(Xq[i0:i0 + chunk], Xc, torch.zeros_like(Xq[i0:i0 + chunk]), nrmc, v2=True)
            P[i0:i0 + chunk] += corr(f) * wc[None, :]
    S = S + 1e-12 * torch.eye(S.shape[0], device=S.device)
    return P @ torch.linalg.solve(S, case["g"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", nargs="+", default=["spot", "cow", "homer", "fandisk"])
    ap.add_argument("--kappa", type=float, default=3.0); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/p2_mesh.json")
    ap.add_argument("--save-correction", default="")
    ap.add_argument("--load-correction", default="")
    ap.add_argument("--train-only", action="store_true")
    a = ap.parse_args()
    if a.save_correction and a.load_correction:
        ap.error("--save-correction and --load-correction are mutually exclusive")
    if a.train_only and not a.save_correction:
        ap.error("--train-only requires --save-correction")
    if a.smoke:
        a.meshes = ["spot"]; a.steps = 40
    torch.set_default_dtype(torch.float64)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} meshes={a.meshes}", flush=True)

    # The correction is trained only on analytic charts. Checkpoint reuse changes
    # execution cost, not the frozen training or evaluation protocol.
    checkpoint_meta = None
    if a.load_correction:
        corr, checkpoint_meta = load_correction(a.load_correction, a.kappa, dev)
    else:
        corr = train_correction(a.kappa, a.steps, a.seed, dev)
        if a.save_correction:
            save_correction(a.save_correction, corr, a.kappa, a.steps, a.seed)
    if a.train_only:
        return

    effective_seed = int(checkpoint_meta["seed"]) if checkpoint_meta is not None else a.seed
    res = {
        "seed": effective_seed,
        "kappa": a.kappa,
        "correction_checkpoint": os.path.abspath(a.load_correction) if a.load_correction else None,
        "source_hashes": source_hashes(),
    }
    for name in a.meshes:
        # decimated dir takes precedence: armadillo/igea/lucy exist full-size in data/ too,
        # and the full-size versions OOM the dense panel assembly
        for sub in ("decimated", "thingi10k", "engineering", "data"):
            path = f"meshes/{sub}/{name}.obj"
            if os.path.exists(path):
                break
        V, F = load_obj(path)
        wt, genus = check_mesh(V, F)
        print(f"[{name}] V={len(V)} F={len(F)} watertight={wt} genus={genus}", flush=True)
        if not wt:
            res[name] = {"watertight": False}; continue
        case = mesh_case(V, F, a.kappa, 900 + effective_seed, dev)
        nq = case["Xq"].shape[0]
        with torch.no_grad():
            # known-kernel validation
            sig0 = torch.linalg.solve(case["Sbase"] + 1e-12 * torch.eye(len(V), device=dev), case["g0"])
            e_known = float(vk.rel(case["Pbase"] @ sig0, case["uq0"]))
            e_floor = float(vk.rel(solve_eval_mesh(case, None), case["uq"]))
            e_corr = float(vk.rel(solve_eval_mesh(case, corr), case["uq"]))
        res[name] = {"V": len(V), "genus": int(genus), "n_probe": int(nq),
                     "tau_h": case["tau_h"],
                     "known_kernel": e_known, "floor": e_floor, "corr_transfer": e_corr}
        print(f"  known-kernel {e_known:.3e} | yukawa floor {e_floor:.3e} -> corr(zero-shot) {e_corr:.3e}"
              f" | tau/h min={case['tau_h']['min']:.2f} p5={case['tau_h']['p5']:.2f}", flush=True)
        del case
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    _mkdir_parent(a.out)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
