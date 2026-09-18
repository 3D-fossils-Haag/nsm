# R geomorph analogues for GMM between sparse LMs, dense LMs, NSM latents

import numpy as np
import pandas as pd

def two_d_array(A):
    """(n, p, k) -> (n, p*k), like geomorph::two.d.array."""
    return np.asarray(A).reshape(len(A), -1)

def mshape(A):
    """Consensus (mean) configuration, geomorph::mshape."""
    return np.asarray(A).mean(axis=0)

def centroid_size(X):
    """Centroid size of one configuration (p, k)."""
    return float(np.sqrt(((X - X.mean(axis=0)) ** 2).sum()))

def procrustes_dist(X, Y):
    """Procrustes distance between two aligned configurations."""
    return float(np.sqrt(((np.asarray(X) - np.asarray(Y)) ** 2).sum()))

def dist_to_mean(A, ref=None):
    """Procrustes distance of every specimen to a reference (default: consensus)."""
    A = np.asarray(A)
    ref = mshape(A) if ref is None else ref
    return np.sqrt(((A - ref) ** 2).sum(axis=(1, 2)))

def gm_prcomp(A):
    """OLS PCA of Procrustes shape variables. Returns geomorph-like fields."""
    A = np.asarray(A, float)
    n_, p_, k_ = A.shape
    X = two_d_array(A)
    mu = X.mean(axis=0)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((S > 1e-10 * S[0]).sum())
    scores = U[:, :rank] * S[:rank]
    eig = (S[:rank] ** 2) / (n_ - 1)
    return {"x": scores, "rotation": Vt[:rank].T, "d": eig,
            "prop": eig / eig.sum(), "cum": np.cumsum(eig / eig.sum()),
            "mean": mu.reshape(p_, k_), "p": p_, "k": k_}

def pc_shape(pca, pc=0, score=None, mag=1.0):
    """Configuration at a given score on a PC (geomorph gm.prcomp$shapes)."""
    if score is None:
        score = pca["x"][:, pc].max()
    v = pca["rotation"][:, pc]
    return (pca["mean"].reshape(-1) + mag * score * v).reshape(pca["p"], pca["k"])

def tps_fit(ref, target, reg=0.0):
    """Solve the 3D TPS that maps ref -> target exactly. ref/target: (p, 3)."""
    ref = np.asarray(ref, float); target = np.asarray(target, float)
    p_ = ref.shape[0]
    K = np.linalg.norm(ref[:, None, :] - ref[None, :, :], axis=2)
    if reg:
        K = K + reg * np.eye(p_)
    P = np.hstack([np.ones((p_, 1)), ref])
    L = np.zeros((p_ + 4, p_ + 4))
    L[:p_, :p_] = K; L[:p_, p_:] = P; L[p_:, :p_] = P.T
    Y = np.vstack([target, np.zeros((4, 3))])
    sol = np.linalg.solve(L, Y)
    return {"ref": ref, "W": sol[:p_], "A": sol[p_:], "K": K}

def tps_apply(tps, pts):
    """Apply a fitted TPS to arbitrary points (m, 3) -- mesh vertices, grid nodes, etc."""
    pts = np.atleast_2d(np.asarray(pts, float))
    Kp = np.linalg.norm(pts[:, None, :] - tps["ref"][None, :, :], axis=2)
    return Kp @ tps["W"] + np.hstack([np.ones((len(pts), 1)), pts]) @ tps["A"]

def morphol_disparity(Y, groups, iter=999, seed=42, partial=False):
    """geomorph::morphol.disparity -- Procrustes variance per group + pairwise permutation test."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    groups = np.asarray(groups)
    lev = np.unique(groups)
    N = len(Y)
    resid = np.zeros_like(Y)
    for l in lev:
        m = groups == l
        resid[m] = Y[m] - Y[m].mean(axis=0)

    def pvar(res, g):
        den = (N - 1) if partial else None
        return np.array([(res[g == l] ** 2).sum() / (den or (g == l).sum()) for l in lev])

    obs = pvar(resid, groups)
    D_obs = np.abs(obs[:, None] - obs[None, :])
    cnt = np.zeros_like(D_obs)
    for _ in range(iter):
        gp = rng.permutation(N)
        pp = pvar(resid[gp], groups)
        cnt += (np.abs(pp[:, None] - pp[None, :]) >= D_obs)
    P = (cnt + 1) / (iter + 1)
    np.fill_diagonal(P, 1.0)
    var_tab = pd.Series(obs, index=lev, name="Procrustes variance")
    return var_tab, pd.DataFrame(D_obs, index=lev, columns=lev), pd.DataFrame(P, index=lev, columns=lev)

def lm_diff(coords, a_idx, b_idx):
    """Per-coordinate difference between two landmarks across all specimens.
    """
    return coords[:, a_idx, :] - coords[:, b_idx, :]

def check_axis_labels(check, axis, values, how, detail=""):
    """Print a per-coordinate statistic and return the coordinate index it selects.
    """
    idx = int(np.argmin(values) if how == "min" else np.argmax(values))
    extreme = "lowest" if how == "min" else "highest"
    print(f"\n{check} - Find which idx is the {axis.upper()}-axis")
    if detail:
        print(detail)
    print(f"idx with the {extreme} value should be {axis.upper()} axis: {axis.upper()} idx = ", idx)
    for i, v in enumerate(values):
        print(f"{i}:", v)
    return idx