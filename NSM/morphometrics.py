# R geomorph analogues
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import boxcox
import re
from pathlib import Path

def two_d_array(A):
    """(n, p, k) -> (n, p*k), like geomorph::two.d.array."""
    return np.asarray(A).reshape(len(A), -1)

def arrayspecs(X, p, k):
    """(n, p*k) -> (n, p, k), like geomorph::arrayspecs."""
    return np.asarray(X).reshape(-1, p, k)

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

def find_mean_spec(A, names=None):
    """geomorph::findMeanSpec -- specimen closest to the consensus shape."""
    d = dist_to_mean(A)
    idx = int(np.argmin(d))
    return (idx, names[idx] if names is not None else idx, d)

def geometric_median(X, tol=1e-10, max_iter=1000):
    """Weiszfeld algorithm: the L1 (geometric) median of the rows of X."""
    X = np.asarray(X, float)
    y = np.median(X, axis=0)
    for _ in range(max_iter):
        d = np.linalg.norm(X - y, axis=1)
        nz = d > 1e-12
        if not nz.any():
            return y
        w = 1.0 / d[nz]
        y_new = (X[nz] * w[:, None]).sum(axis=0) / w.sum()
        n_zero = int((~nz).sum())
        if n_zero:                                   # y sits exactly on a data point
            r = np.linalg.norm(((X[nz] - y) * w[:, None]).sum(axis=0))
            rinv = 0.0 if r == 0 else n_zero / r
            y_new = max(0.0, 1 - rinv) * y_new + min(1.0, rinv) * y
        if np.linalg.norm(y_new - y) < tol:
            return y_new
        y = y_new
    return y

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


def scree_plot(pca, n_show=20, outfpath=None):
    n_show = min(n_show, len(pca["prop"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(np.arange(1, n_show + 1), 100 * pca["prop"][:n_show], color="#34495e")
    ax.set_xlabel("Principal component"); ax.set_ylabel("% variance")
    ax2 = ax.twinx()
    ax2.plot(np.arange(1, n_show + 1), 100 * pca["cum"][:n_show], "o-", color="#c0392b", ms=3)
    ax2.set_ylabel("cumulative %", color="#c0392b"); ax2.set_ylim(0, 101)
    n95 = int(np.searchsorted(pca["cum"], 0.95) + 1)
    ax.set_title(f"Shape-space PCA — {n95} PCs reach 95% of variance")
    plt.tight_layout()
    if outfpath: plt.savefig(outfpath, dpi=300, bbox_inches="tight")
    plt.show()

def find_median_spec(A, kind="geometric", names=None):
    """Specimen closest to the median shape (no direct geomorph equivalent).

    kind = 'geometric' : L1 / geometric median of the shape vectors (recommended --
                         a true multivariate median, robust to outlier specimens)
           'coordwise' : per-coordinate median (fast, but not a shape in general)
           'dist'      : the specimen with the median Procrustes distance from the consensus
    """
    A = np.asarray(A)
    n_, p_, k_ = A.shape
    X = two_d_array(A)
    if kind == "dist":
        d = dist_to_mean(A)
        idx = int(np.argsort(d)[n_ // 2])
        return idx, (names[idx] if names is not None else idx), A[idx], d
    med = geometric_median(X) if kind == "geometric" else np.median(X, axis=0)
    d = np.linalg.norm(X - med, axis=1)
    idx = int(np.argmin(d))
    return idx, (names[idx] if names is not None else idx), med.reshape(p_, k_), d

def outlier_limit(d):
    """Power-transform distances, take Tukey's Q3 + 1.5*IQR fence, back-transform."""
    d = np.asarray(d, float)
    shift = d.min() - 1e-9
    dpos = d - shift
    try:
        dt, lam = boxcox(dpos)
    except Exception:
        dt, lam = dpos, 1.0
    q1, q3 = np.percentile(dt, [25, 75])
    lim_t = q3 + 1.5 * (q3 - q1)
    if np.isclose(lam, 0):
        lim = np.exp(lim_t)
    else:
        base = lam * lim_t + 1
        lim = base ** (1 / lam) if base > 0 else np.inf
    return lim + shift, lam


def plot_outliers(A, names, groups=None, PC=None, pca=None, title="", n_label=12,
                  figsize=(9, 5), outfpath=None):
    """Specimens ordered by distance from the mean; those past the Tukey fence in red.

    Reads like geomorph's plotOutliers: rank on x, distance on y, dashed fence line.
    n_label : how many of the most extreme specimens to annotate (0 = none).
    PC      : optional sequence of 0-based PC indices -- screen in a shape subspace only.
    """
    A = np.asarray(A)
    names = list(names)
    if PC is not None:
        if pca is None:
            pca = gm_prcomp(A)
        scores = pca["x"][:, list(PC)]
        d_all = np.linalg.norm(scores - scores.mean(axis=0), axis=1)
        ylab = f"distance in PC{[i+1 for i in PC]} space"
    else:
        d_all = dist_to_mean(A)
        ylab = "Procrustes distance from mean"

    def _one(idx_subset, sub_title, ax=None):
        idx_subset = np.asarray(idx_subset)
        dd = d_all[idx_subset]
        order = np.argsort(dd)                       # ascending, as geomorph plots it
        lim, lam = outlier_limit(dd)
        flagged = [int(idx_subset[i]) for i in order[::-1] if dd[i] > lim]

        own_fig = ax is None
        if own_fig:
            fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(order))
        cols = np.where(dd[order] > lim, "#c0392b", "#2c3e50")
        ax.scatter(x, dd[order], c=cols, s=12, linewidths=0)
        ax.axhline(lim, ls="--", c="#c0392b", lw=1)
        ax.text(0, lim, f" Tukey fence = {lim:.4f}", color="#c0392b", fontsize=8,
                va="bottom", ha="left")
        ax.set_xlabel("specimen (ordered by distance)")
        ax.set_ylabel(ylab)
        ax.set_title(f"{sub_title} — {len(flagged)} of {len(dd)} flagged "
                     f"(Box-Cox lambda={lam:.2f})", fontsize=10)
        ax.margins(x=0.02)

        span = float(dd.max() - dd.min()) or 1.0
        last_y = None
        for rank, i in enumerate(order[::-1][:n_label]):
            if dd[i] <= lim:
                break
            crowded = (last_y is not None and abs(last_y - dd[i]) < 0.04 * span)
            if crowded or (dd[i] - lim) < 0.05 * span:
                continue                      # skip labels that would overlap each other or the fence
            ax.annotate(names[idx_subset[i]], (len(order) - 1 - rank, dd[i]),
                        textcoords="offset points", xytext=(-6, 0), ha="right",
                        fontsize=6.5, color="#c0392b", va="center")
            last_y = dd[i]
        if own_fig:
            plt.tight_layout()
            if outfpath:
                plt.savefig(outfpath, dpi=300, bbox_inches="tight")
            plt.show()
        return flagged

    if groups is None:
        return _one(np.arange(len(A)), title or "Potential outliers")

    glev = [g for g in pd.unique(pd.Series(groups))
            if (np.asarray(groups) == g).sum() >= 4]
    ncol = min(3, len(glev))
    nrow = int(np.ceil(len(glev) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize[0] / 2.2 * ncol, figsize[1] * 0.85 * nrow),
                             squeeze=False)
    out = {}
    for ax, g in zip(axes.ravel(), glev):
        out[g] = _one(np.where(np.asarray(groups) == g)[0], str(g), ax=ax)
    for ax in axes.ravel()[len(glev):]:
        ax.axis("off")
    fig.suptitle(title or "Potential outliers by group", fontsize=11)
    plt.tight_layout()
    if outfpath:
        plt.savefig(outfpath, dpi=300, bbox_inches="tight")
    plt.show()
    return out

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

def bending_energy(tps):
    """Bending energy of a warp: 0 for a purely affine (uniform) difference."""
    W = tps["W"]
    return float(-np.trace(W.T @ tps["K"] @ W))   # sign convention for U(r)=r in 3D

def plot_tps_grid(ref, target, planes=((0, 1), (0, 2)), n_grid=22, mag=1.0, pad=0.12,
                  axis_names=None, show_points=True, links=None, title="",
                  figsize=None, outfpath=None):
    """geomorph::plotRefToTarget(method='TPS').

    For 3D data geomorph draws the spline in the x-y and x-z planes; add (1, 2) for y-z.
    mag magnifies the reference->target difference (geomorph's `mag`).
    """
    ref = np.asarray(ref, float)
    target = np.asarray(target, float)
    tgt_mag = ref + mag * (target - ref)
    tps = tps_fit(ref, tgt_mag)
    be = bending_energy(tps)

    planes = [(a, b) for a, b in planes if a != b]
    if not planes:
        raise ValueError("planes must use two different coordinate indices")
    fig, axes = plt.subplots(1, len(planes), figsize=figsize or (6 * len(planes), 5.5))
    axes = np.atleast_1d(axes)
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    rng = np.where((hi - lo) > 0, hi - lo, 1.0)

    for ax, (a, b) in zip(axes, planes):
        c = ({0, 1, 2} - {a, b}).pop()
        ga = np.linspace(lo[a] - pad * rng[a], hi[a] + pad * rng[a], n_grid)
        gb = np.linspace(lo[b] - pad * rng[b], hi[b] + pad * rng[b], n_grid)
        GA, GB = np.meshgrid(ga, gb, indexing="ij")
        pts = np.zeros((GA.size, 3))
        pts[:, a] = GA.ravel(); pts[:, b] = GB.ravel(); pts[:, c] = ref[:, c].mean()
        W = tps_apply(tps, pts)
        WA = W[:, a].reshape(n_grid, n_grid); WB = W[:, b].reshape(n_grid, n_grid)

        for i in range(n_grid):
            ax.plot(WA[i, :], WB[i, :], color="#95a5a6", lw=0.6, zorder=1)
            ax.plot(WA[:, i], WB[:, i], color="#95a5a6", lw=0.6, zorder=1)
        if links is not None:
            for i, j in links:
                ax.plot([tgt_mag[i, a], tgt_mag[j, a]], [tgt_mag[i, b], tgt_mag[j, b]],
                        color="#2c3e50", lw=1.0, zorder=2)
        if show_points:
            ax.scatter(ref[:, a], ref[:, b], s=22, facecolors="none",
                       edgecolors="#7f8c8d", lw=0.8, zorder=3, label="reference")
            ax.scatter(tgt_mag[:, a], tgt_mag[:, b], s=22, color="#c0392b",
                       zorder=4, label="target")
            for i in range(len(ref)):
                ax.annotate("", xy=tgt_mag[i, [a, b]], xytext=ref[i, [a, b]],
                            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7, alpha=0.7),
                            zorder=3)
        ax.set_xlabel(axis_names[a]); ax.set_ylabel(axis_names[b])
        ax.set_title(f"{str(axis_names[a]).split()[0]}–{str(axis_names[b]).split()[0]} plane")
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"{title}   (mag={mag}g, bending energy={be:.3e})", fontsize=11)
    plt.tight_layout()
    if outfpath: plt.savefig(outfpath, dpi=300, bbox_inches="tight")
    plt.show()
    return tps

def factor_to_bool(f):
    """Treatment-coded design block for a factor (first level = reference)."""
    f = np.asarray(f)
    lev = pd.unique(pd.Series(f))
    lev = np.sort(lev)
    return np.column_stack([(f == l).astype(float) for l in lev[1:]]), lev

def interaction_block(D1, D2):
    return np.column_stack([D1[:, i] * D2[:, j] for i in range(D1.shape[1]) for j in range(D2.shape[1])])

def procD_lm(Y, terms, iter=999, seed=42, verbose=True):
    """Procrustes ANOVA with residual randomization (geomorph::procD.lm, RRPP).

    Y     : (n, m) shape variables -- use two_d_array(coords) or PC scores.
    terms : ordered list of (name, design_block) for sequential (Type I) SS.
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    n_ = len(Y)
    blocks = [np.ones((n_, 1))] + [np.asarray(b, float) for _, b in terms]
    names = [nm for nm, _ in terms]
    designs = [np.hstack(blocks[:i + 1]) for i in range(len(blocks))]
    ranks = [np.linalg.matrix_rank(X) for X in designs]

    def fit_res(X, Yv):
        B, *_ = np.linalg.lstsq(X, Yv, rcond=None)
        return Yv - X @ B

    def ss(r): return float((r ** 2).sum())

    res_full = [fit_res(X, Y) for X in designs]
    SS_obs = [ss(res_full[i]) - ss(res_full[i + 1]) for i in range(len(names))]
    df = [ranks[i + 1] - ranks[i] for i in range(len(names))]
    SSE, dfE = ss(res_full[-1]), n_ - ranks[-1]
    SST = ss(res_full[0])
    F_obs = [(SS_obs[i] / df[i]) / (SSE / dfE) for i in range(len(names))]

    fitted_red = [Y - res_full[i] for i in range(len(names))]
    counts = np.zeros(len(names))
    for _ in range(iter):
        perm = rng.permutation(n_)
        for i in range(len(names)):
            Yr = fitted_red[i] + res_full[i][perm]           # RRPP: randomise reduced-model residuals
            ssi = ss(fit_res(designs[i], Yr)) - ss(fit_res(designs[i + 1], Yr))
            sse = ss(fit_res(designs[-1], Yr))
            counts[i] += ((ssi / df[i]) / (sse / dfE)) >= F_obs[i]
    pvals = (counts + 1) / (iter + 1)

    tab = pd.DataFrame({"term": names, "Df": df, "SS": SS_obs,
                        "MS": [SS_obs[i] / df[i] for i in range(len(names))],
                        "Rsq": [SS_obs[i] / SST for i in range(len(names))],
                        "F": F_obs, "Pr(>F)": pvals})
    tab.loc[len(tab)] = ["Residuals", dfE, SSE, SSE / dfE, SSE / SST, np.nan, np.nan]
    tab.loc[len(tab)] = ["Total", n_ - 1, SST, np.nan, 1.0, np.nan, np.nan]
    if verbose:
        print(f"Procrustes ANOVA (RRPP, {iter} permutations)")
    return tab

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

def read_newick(path_or_string):
    """Minimal Newick/NEXUS reader. Handles missing branch lengths and polytomies."""
    s = str(path_or_string)
    if Path(s).exists():
        s = Path(s).read_text()
    s = re.sub(r"\[[^\]]*\]", "", s).strip()
    if s.lower().startswith("#nexus"):
        m = re.search(r"(?im)^\s*tree\s+[^=]+=\s*(\(.*?;)", s, re.S)
        if not m:
            raise ValueError("No tree line found in the NEXUS file")
        s = m.group(1)
    s = s.strip()
    if not s.endswith(";"):
        s += ";"
    nodes = []

    def new_node(parent):
        nodes.append({"parent": parent, "children": [], "name": None, "length": None})
        if parent is not None:
            nodes[parent]["children"].append(len(nodes) - 1)
        return len(nodes) - 1

    i, root = 0, new_node(None)
    cur = root
    while i < len(s):
        ch = s[i]
        if ch == "(":
            cur = new_node(cur); i += 1
        elif ch == ",":
            cur = new_node(nodes[cur]["parent"]); i += 1
        elif ch == ")":
            cur = nodes[cur]["parent"]; i += 1
        elif ch == ";":
            break
        elif ch in " \t\r\n":
            i += 1
        elif ch == ":":
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] in ".eE+-"):
                j += 1
            nodes[cur]["length"] = float(s[i + 1:j]); i = j
        elif ch in "\'\"":
            j = s.index(ch, i + 1)
            nodes[cur]["name"] = s[i + 1:j]; i = j + 1
        else:
            j = i
            while j < len(s) and s[j] not in "(),:;":
                j += 1
            nodes[cur]["name"] = s[i:j].strip(); i = j
    return {"nodes": nodes, "root": root}

def is_tip(tree, i):      return len(tree["nodes"][i]["children"]) == 0
def tip_indices(tree):    return [i for i in range(len(tree["nodes"])) if is_tip(tree, i)]
def tip_names(tree):      return [tree["nodes"][i]["name"] for i in tip_indices(tree)]
def has_brlen(tree):
    return all(nd["length"] is not None for nd in tree["nodes"] if nd["parent"] is not None)

def _rebuild(nodes, root):
    order, stack = [], [root]
    while stack:
        i = stack.pop(); order.append(i); stack.extend(nodes[i]["children"])
    remap = {old: new for new, old in enumerate(order)}
    out = [{"parent": None if nodes[o]["parent"] is None else remap.get(nodes[o]["parent"]),
            "children": [remap[c] for c in nodes[o]["children"]],
            "name": nodes[o]["name"], "length": nodes[o]["length"]} for o in order]
    return {"nodes": out, "root": 0}

def prune_tree(tree, keep):
    """Keep the listed tips, drop empty clades, collapse single-child nodes (ape::drop.tip)."""
    keep = set(keep)
    nodes = [dict(nd, children=list(nd["children"])) for nd in tree["nodes"]]

    def keeps(i):
        if not nodes[i]["children"]:
            return nodes[i]["name"] in keep
        nodes[i]["children"] = [c for c in nodes[i]["children"] if keeps(c)]
        return len(nodes[i]["children"]) > 0

    keeps(tree["root"])

    def collapse(i):
        for c in list(nodes[i]["children"]):
            collapse(c)
        ch = nodes[i]["children"]
        if len(ch) == 1 and nodes[i]["parent"] is not None:
            c, par = ch[0], nodes[i]["parent"]
            nodes[par]["children"] = [c if x == i else x for x in nodes[par]["children"]]
            nodes[c]["parent"] = par
            if nodes[c]["length"] is not None and nodes[i]["length"] is not None:
                nodes[c]["length"] += nodes[i]["length"]

    collapse(tree["root"])
    root = tree["root"]
    while len(nodes[root]["children"]) == 1:
        root = nodes[root]["children"][0]
        nodes[root]["parent"] = None
    return _rebuild(nodes, root)

def resolve_polytomies(tree, seed=42):
    """ape::multi2di -- random dichotomous resolution, new branches of length 0."""
    rng = np.random.default_rng(seed)
    nodes = [dict(nd, children=list(nd["children"])) for nd in tree["nodes"]]
    changed = True
    while changed:
        changed = False
        for i in range(len(nodes)):
            ch = nodes[i]["children"]
            if len(ch) > 2:
                a, b = [ch[j] for j in rng.choice(len(ch), size=2, replace=False)]
                nodes.append({"parent": i, "children": [a, b], "name": None, "length": 0.0})
                new = len(nodes) - 1
                nodes[a]["parent"] = nodes[b]["parent"] = new
                nodes[i]["children"] = [c for c in ch if c not in (a, b)] + [new]
                changed = True
    return _rebuild(nodes, tree["root"])

def grafen_brlen(tree, power=1.0):
    """ape::compute.brlen(method='Grafen') -- node height from the number of descendant tips."""
    nodes = [dict(nd, children=list(nd["children"])) for nd in tree["nodes"]]
    n_desc = [0] * len(nodes)

    def count(i):
        n_desc[i] = 1 if not nodes[i]["children"] else sum(count(c) for c in nodes[i]["children"])
        return n_desc[i]

    count(tree["root"])
    N = n_desc[tree["root"]]
    h = [0.0 if not nodes[i]["children"] else (n_desc[i] - 1) / (N - 1) for i in range(len(nodes))]
    for i in range(len(nodes)):
        par = nodes[i]["parent"]
        nodes[i]["length"] = 0.0 if par is None else (h[par] ** power - h[i] ** power)
    return {"nodes": nodes, "root": tree["root"]}

def vcv(tree, nodes_too=False):
    """Brownian covariance among tips: shared root-to-MRCA path length (ape::vcv)."""
    paths = {}

    def walk(i, path):
        path = path + [i]
        paths[i] = path
        for c in tree["nodes"][i]["children"]:
            walk(c, path)

    walk(tree["root"], [])
    L = [0.0 if nd["length"] is None else nd["length"] for nd in tree["nodes"]]
    tips = tip_indices(tree)
    names = [tree["nodes"][i]["name"] for i in tips]

    def shared(pi, pj):
        s = 0.0
        for a in range(min(len(pi), len(pj))):
            if pi[a] != pj[a]:
                break
            s += L[pi[a]]
        return s

    nt = len(tips)
    C = np.zeros((nt, nt))
    for a in range(nt):
        for b in range(a, nt):
            C[a, b] = C[b, a] = shared(paths[tips[a]], paths[tips[b]])
    if not nodes_too:
        return names, C
    internal = [i for i in range(len(tree["nodes"])) if not is_tip(tree, i)]
    Cnt = np.array([[shared(paths[m], paths[t]) for t in tips] for m in internal])
    return names, C, internal, Cnt

def phylo_mean(Y, Cinv):
    one = np.ones((len(Y), 1))
    return np.linalg.solve(one.T @ Cinv @ one, one.T @ Cinv @ Y)

def physignal(Y, C, iter=999, seed=42):
    """geomorph::physignal -- Adams (2014) Kmult with a permutation test.

    K near 1 means the amount of covariation matches Brownian expectation on this tree;
    K well below 1 means less similarity among relatives than Brownian motion predicts.
    The P-value comes from permuting shapes across tips.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    n = len(Y)
    rng = np.random.default_rng(seed)
    Cinv = np.linalg.inv(C)
    E = (np.trace(C) - n / Cinv.sum()) / (n - 1)

    def K_of(Yv):
        a = phylo_mean(Yv, Cinv)
        D = Yv - a
        return (np.trace(D.T @ D) / np.trace(D.T @ Cinv @ D)) / E

    K_obs = K_of(Y)
    perm = np.array([K_of(Y[rng.permutation(n)]) for _ in range(iter)])
    return {"K": float(K_obs), "p": float((np.sum(perm >= K_obs) + 1) / (iter + 1)),
            "perm": perm, "n": n}

def _gls_whitener(C):
    """C^(-1/2), used to transform the data into an independent-error space."""
    w, V = np.linalg.eigh(C)
    return V @ np.diag(np.clip(w, 1e-12, None) ** -0.5) @ V.T

def procD_lm_general(Y, terms, iter=999, seed=42, transform=None, verbose=True):
    """Sequential SS with RRPP; `transform` applies GLS whitening (PGLS) when supplied."""
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, float)
    n_ = len(Y)
    names = [nm for nm, _ in terms]
    B = [np.ones((n_, 1))] + [np.asarray(b, float) for _, b in terms]
    if transform is not None:
        Y = transform @ Y
        B = [transform @ b for b in B]
    designs = [np.hstack(B[:i + 1]) for i in range(len(B))]
    ranks = [np.linalg.matrix_rank(X) for X in designs]

    def res(X, Yv):
        beta, *_ = np.linalg.lstsq(X, Yv, rcond=None)
        return Yv - X @ beta

    def ss(r): return float((r ** 2).sum())

    rf = [res(X, Y) for X in designs]
    SS = [ss(rf[i]) - ss(rf[i + 1]) for i in range(len(names))]
    df = [ranks[i + 1] - ranks[i] for i in range(len(names))]
    SSE, dfE, SST = ss(rf[-1]), n_ - ranks[-1], ss(rf[0])
    F = [(SS[i] / df[i]) / (SSE / dfE) for i in range(len(names))]

    fitted = [Y - rf[i] for i in range(len(names))]
    cnt = np.zeros(len(names))
    for _ in range(iter):
        pm = rng.permutation(n_)
        for i in range(len(names)):
            Yr = fitted[i] + rf[i][pm]
            si = ss(res(designs[i], Yr)) - ss(res(designs[i + 1], Yr))
            se = ss(res(designs[-1], Yr))
            cnt[i] += ((si / df[i]) / (se / dfE)) >= F[i]
    pv = (cnt + 1) / (iter + 1)

    tab = pd.DataFrame({"term": names, "Df": df, "SS": SS,
                        "MS": [SS[i] / df[i] for i in range(len(names))],
                        "Rsq": [s / SST for s in SS], "F": F, "Pr(>F)": pv})
    tab.loc[len(tab)] = ["Residuals", dfE, SSE, SSE / dfE, SSE / SST, np.nan, np.nan]
    tab.loc[len(tab)] = ["Total", n_ - 1, SST, np.nan, 1.0, np.nan, np.nan]
    if verbose:
        kind = "PGLS" if transform is not None else "OLS"
        print(f"Procrustes ANOVA ({kind}, RRPP, {iter} permutations)")
    return tab

def procD_pgls(Y, terms, C, iter=999, seed=42, verbose=True):
    """geomorph::procD.pgls -- Procrustes ANOVA with a Brownian error structure."""
    return procD_lm_general(Y, terms, iter=iter, seed=seed,
                            transform=_gls_whitener(C), verbose=verbose)