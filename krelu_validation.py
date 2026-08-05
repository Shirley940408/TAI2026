"""
Numerical validation of the k-ReLU joint relaxation used in verifier.py.

For a group G of k ReLU neurons whose pre-activations z_j are affine in the
input, we build

    P = box(l, u)  intersect  octahedral constraints on z_a +/- z_b

and enumerate the vertices of P subdivided by the hyperplanes z_j = 0.  The
convex hull of {(z, ReLU(z)) : z in P} has all its extreme points among
{(v, ReLU(v)) : v such a vertex}, so for any d

    sum_j c_j y_j  >=  sum_j d_j z_j + e,
    e = min_v ( sum_j c_j ReLU(v_j) - sum_j d_j v_j )

is a valid inequality.  Substituting the symbolic z_j and concretising over
the input box gives a bound that is optimised over d.

Checked here:
  1. the refined bound never exceeds the true minimum (soundness),
  2. at the DeepPoly-equivalent initial d it is never worse than DeepPoly,
  3. optimising d strictly improves it in practice.
"""
import itertools

import numpy as np

TOL = 1e-9


def plane_system(k):
    """Subdivision planes and constraints for a group of size k."""
    eye = np.eye(k)
    pairs = list(itertools.combinations(range(k), 2))

    normals = []           # plane normals, in a fixed order
    for a in range(k):     # z_a = l_a
        normals.append(eye[a])
    for a in range(k):     # z_a = u_a
        normals.append(eye[a])
    for a in range(k):     # z_a = 0
        normals.append(eye[a])
    for a, b in pairs:     # z_a + z_b = lo / hi, z_a - z_b = lo / hi
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] - eye[b])
        normals.append(eye[a] - eye[b])
    normals = np.array(normals)

    con_normals = []       # constraints written as  n . z <= rhs
    for a in range(k):
        con_normals.append(eye[a])
        con_normals.append(-eye[a])
    for a, b in pairs:
        con_normals.append(eye[a] + eye[b])
        con_normals.append(-(eye[a] + eye[b]))
        con_normals.append(eye[a] - eye[b])
        con_normals.append(-(eye[a] - eye[b]))
    con_normals = np.array(con_normals)

    combos = [c for c in itertools.combinations(range(len(normals)), k)]
    keep = []
    inverses = []
    for c in combos:
        M = normals[list(c)]
        if abs(np.linalg.det(M)) > 1e-8:
            keep.append(c)
            inverses.append(np.linalg.inv(M))
    return normals, con_normals, np.array(keep), np.array(inverses), pairs


def group_geometry(l, u, sum_lo, sum_hi, diff_lo, diff_hi, k, pairs):
    """Plane offsets and constraint right-hand sides for one group.

    The right-hand sides must follow exactly the same order as the rows of
    `con_normals` in plane_system: (+e_a, -e_a) interleaved per axis, then
    four entries per pair.
    """
    offsets = list(l) + list(u) + [0.0] * k
    rhs = []
    for a in range(k):
        rhs += [u[a], -l[a]]
    for p, (a, b) in enumerate(pairs):
        offsets += [sum_lo[p], sum_hi[p], diff_lo[p], diff_hi[p]]
        rhs += [sum_hi[p], -sum_lo[p], diff_hi[p], -diff_lo[p]]
    return np.array(offsets), np.array(rhs)


def vertices(offsets, rhs, con_normals, keep, inverses, tol=1e-7):
    pts = np.einsum('cij,cj->ci', inverses, offsets[keep])
    ok = np.all(pts @ con_normals.T <= rhs[None, :] + tol, axis=1)
    return pts[ok]


def concretize_lower(a, const, x_l, x_u):
    return np.maximum(a, 0) @ x_l + np.minimum(a, 0) @ x_u + const


def run_trial(seed, k=3, n_in=5, verbose=False):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(k, n_in))
    b = rng.normal(size=k) * 0.5
    x0 = rng.random(n_in)
    eps = 0.45
    x_l = np.clip(x0 - eps, 0, 1)
    x_u = np.clip(x0 + eps, 0, 1)

    l = np.array([concretize_lower(W[j], b[j], x_l, x_u) for j in range(k)])
    u = np.array([-concretize_lower(-W[j], -b[j], x_l, x_u) for j in range(k)])
    if not np.all((l < 0) & (u > 0)):
        return None  # only interesting when every neuron is unstable

    normals, con_normals, keep, inverses, pairs = plane_system(k)

    sum_lo, sum_hi, diff_lo, diff_hi = [], [], [], []
    for a, bb in pairs:
        s = W[a] + W[bb]
        sc = b[a] + b[bb]
        sum_lo.append(max(concretize_lower(s, sc, x_l, x_u), l[a] + l[bb]))
        sum_hi.append(min(-concretize_lower(-s, -sc, x_l, x_u), u[a] + u[bb]))
        dd = W[a] - W[bb]
        dc = b[a] - b[bb]
        diff_lo.append(max(concretize_lower(dd, dc, x_l, x_u), l[a] - u[bb]))
        diff_hi.append(min(-concretize_lower(-dd, -dc, x_l, x_u), u[a] - l[bb]))

    offsets, rhs = group_geometry(l, u, sum_lo, sum_hi, diff_lo, diff_hi, k, pairs)
    V = vertices(offsets, rhs, con_normals, keep, inverses)
    if V.shape[0] == 0:
        return None

    c = rng.normal(size=k)

    # ---- DeepPoly reference ------------------------------------------
    lam = np.where(u >= -l, 1.0, 0.0)          # 0/1 area heuristic
    up_slope = u / (u - l)
    up_int = -up_slope * l
    dp_a = np.zeros(n_in)
    dp_c = 0.0
    for j in range(k):
        if c[j] > 0:
            dp_a += c[j] * lam[j] * W[j]
            dp_c += c[j] * lam[j] * b[j]
        else:
            dp_a += c[j] * up_slope[j] * W[j]
            dp_c += c[j] * (up_slope[j] * b[j] + up_int[j])
    dp_bound = concretize_lower(dp_a, dp_c, x_l, x_u)

    # ---- k-ReLU with the DeepPoly-equivalent d ------------------------
    def krelu_bound(d):
        e = np.min(c @ np.maximum(V, 0).T - d @ V.T)
        a = np.zeros(n_in)
        const = e
        for j in range(k):
            a += d[j] * W[j]
            const += d[j] * b[j]
        return concretize_lower(a, const, x_l, x_u)

    d_init = np.where(c > 0, c * lam, c * up_slope)
    init_bound = krelu_bound(d_init)

    # ---- optimise d by projected subgradient ascent -------------------
    d = d_init.copy()
    best = init_bound
    step = 0.5 * (np.abs(c).max() + 1e-6)
    for it in range(400):
        e_terms = c @ np.maximum(V, 0).T - d @ V.T
        v_star = V[int(np.argmin(e_terms))]
        a = d @ W
        grad_e = -v_star
        # subgradient of concretize_lower(a, .) wrt d
        x_star = np.where(a > 0, x_l, x_u)
        grad_a = W @ x_star
        g = grad_a + grad_e
        d = d + (step / np.sqrt(it + 1)) * g
        val = krelu_bound(d)
        if val > best:
            best = val
    opt_bound = best

    # ---- ground truth -------------------------------------------------
    X = x_l + (x_u - x_l) * rng.random((60000, n_in))
    Z = X @ W.T + b
    true_min = float(np.min(np.maximum(Z, 0) @ c))

    return dict(dp=dp_bound, init=init_bound, opt=opt_bound,
                truth=true_min, nverts=V.shape[0])


def main():
    for k in (2, 3):
        rows = []
        problems = []
        seed = 0
        while len(rows) < 40:
            r = run_trial(seed, k=k)
            seed += 1
            if r is None:
                continue
            rows.append(r)
            if r['dp'] > r['truth'] + 1e-6:
                problems.append(f"seed {seed}: DeepPoly {r['dp']:.6f} > truth {r['truth']:.6f}")
            if r['init'] > r['truth'] + 1e-6:
                problems.append(f"seed {seed}: kReLU-init {r['init']:.6f} > truth {r['truth']:.6f}")
            if r['opt'] > r['truth'] + 1e-6:
                problems.append(f"seed {seed}: kReLU-opt {r['opt']:.6f} > truth {r['truth']:.6f}")
            if r['init'] < r['dp'] - 1e-9:
                problems.append(f"seed {seed}: kReLU-init {r['init']:.6f} < DeepPoly {r['dp']:.6f}")

        dp = np.array([r['dp'] for r in rows])
        init = np.array([r['init'] for r in rows])
        opt = np.array([r['opt'] for r in rows])
        truth = np.array([r['truth'] for r in rows])
        verts = np.array([r['nverts'] for r in rows])

        gap_dp = truth - dp
        gap_opt = truth - opt
        closed = 1.0 - gap_opt / np.maximum(gap_dp, 1e-9)

        print(f'--- k = {k} ---')
        print(f'  vertices per group: mean {verts.mean():.1f}  max {verts.max()}')
        print(f'  mean gap to truth: DeepPoly {gap_dp.mean():.4f} '
              f'-> k-ReLU {gap_opt.mean():.4f}')
        print(f'  fraction of the DeepPoly gap closed: mean {closed.mean():.1%}  '
              f'median {np.median(closed):.1%}')
        print(f'  improvement over DeepPoly: mean {(opt - dp).mean():.4f}  '
              f'max {(opt - dp).max():.4f}')
        if problems:
            print(f'  SOUNDNESS PROBLEMS: {len(problems)}')
            for p in problems[:8]:
                print('    ', p)
        else:
            print('  no violations (bounds always below truth, never below DeepPoly)')
        print()


if __name__ == '__main__':
    main()