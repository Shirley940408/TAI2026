"""
NumPy shadow of code/krelu.py, mirroring it operation for operation so the
exact tensor logic (index orders, masks, matmul folding) can be executed and
tested here, where torch is unavailable.

Any behavioural difference between this file and code/krelu.py is a bug in
one of them.
"""
import itertools

import numpy as np

_VERTEX_REL_TOL = 1e-4
_VERTEX_ABS_TOL = 1e-6

_PLANE_CACHE = {}


def plane_system(k):
    if k in _PLANE_CACHE:
        return _PLANE_CACHE[k]

    eye = np.eye(k)
    pairs = list(itertools.combinations(range(k), 2))

    normals = []
    for a in range(k):
        normals.append(eye[a])
    for a in range(k):
        normals.append(eye[a])
    for a in range(k):
        normals.append(eye[a])
    for a, b in pairs:
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] - eye[b])
        normals.append(eye[a] - eye[b])
    normals = np.stack(normals)

    cons = []
    for a in range(k):
        cons.append(eye[a])
        cons.append(-eye[a])
    for a, b in pairs:
        cons.append(eye[a] + eye[b])
        cons.append(-(eye[a] + eye[b]))
        cons.append(eye[a] - eye[b])
        cons.append(-(eye[a] - eye[b]))
    cons = np.stack(cons)

    keep, inverses = [], []
    for combo in itertools.combinations(range(normals.shape[0]), k):
        M = normals[list(combo)]
        if abs(np.linalg.det(M)) > 1e-8:
            keep.append(combo)
            inverses.append(np.linalg.inv(M))

    out = dict(normals=normals, cons=cons, combos=np.array(keep),
               inverses=np.stack(inverses), pairs=pairs, k=k)
    _PLANE_CACHE[k] = out
    return out


def _lower(A, b, x_l, x_u):
    return np.maximum(A, 0) @ x_l + np.minimum(A, 0) @ x_u + b


def _upper(A, b, x_l, x_u):
    return np.maximum(A, 0) @ x_u + np.minimum(A, 0) @ x_l + b


def pair_bounds(ia, ib, A_low, b_low, A_high, b_high, l, u, x_l, x_u):
    As = A_low[ia] + A_low[ib]
    bs = b_low[ia] + b_low[ib]
    sum_lo = np.maximum(_lower(As, bs, x_l, x_u), l[ia] + l[ib])

    Ash = A_high[ia] + A_high[ib]
    bsh = b_high[ia] + b_high[ib]
    sum_hi = np.minimum(_upper(Ash, bsh, x_l, x_u), u[ia] + u[ib])

    Ad = A_low[ia] - A_high[ib]
    bd = b_low[ia] - b_high[ib]
    diff_lo = np.maximum(_lower(Ad, bd, x_l, x_u), l[ia] - u[ib])

    Adh = A_high[ia] - A_low[ib]
    bdh = b_high[ia] - b_low[ib]
    diff_hi = np.minimum(_upper(Adh, bdh, x_l, x_u), u[ia] - l[ib])
    return sum_lo, sum_hi, diff_lo, diff_hi


def group_vertices(groups, A_low, b_low, A_high, b_high, l, u, x_l, x_u):
    G, k = groups.shape
    system = plane_system(k)
    pairs = system['pairs']

    lg = l[groups]
    ug = u[groups]

    offsets = [lg, ug, np.zeros_like(lg)]
    rhs = []
    for a in range(k):
        rhs.append(ug[:, a])
        rhs.append(-lg[:, a])

    for a, b in pairs:
        s_lo, s_hi, d_lo, d_hi = pair_bounds(
            groups[:, a], groups[:, b],
            A_low, b_low, A_high, b_high, l, u, x_l, x_u)
        offsets.append(np.stack([s_lo, s_hi, d_lo, d_hi], axis=1))
        rhs.extend([s_hi, -s_lo, d_hi, -d_lo])

    offsets = np.concatenate(offsets, axis=1)
    rhs = np.stack(rhs, axis=1)

    selected = offsets[:, system['combos']]
    verts = np.einsum('cij,gcj->gci', system['inverses'], selected)

    lhs = np.einsum('nk,gck->gcn', system['cons'], verts)
    tol = _VERTEX_ABS_TOL + _VERTEX_REL_TOL * np.abs(rhs).max(axis=1, keepdims=True)
    feasible = np.all(lhs <= rhs[:, None, :] + tol[:, :, None], axis=2)
    return verts, feasible


def group_constants(c_rows, groups, verts, feasible, d):
    if not np.all(feasible.any(axis=1)):
        return None
    cg = c_rows[:, groups]
    relu_v = np.maximum(verts, 0)
    term = np.einsum('mgk,gck->mgc', cg, relu_v) - np.einsum('mgk,gck->mgc', d, verts)
    term = np.where(feasible[None, :, :], term, np.inf)
    return term.min(axis=2).sum(axis=1)


def deeppoly_d(c_rows, groups, lower_scale, upper_scale):
    cg = c_rows[:, groups]
    ls = lower_scale[groups][None]
    us = upper_scale[groups][None]
    return np.where(cg > 0, cg * ls, cg * us)


def refine_linear_forms(c_rows, const, groups, d, tail, x_l, x_u):
    A_low, b_low = tail['A_low'], tail['b_low']
    A_high, b_high = tail['A_high'], tail['b_high']
    lower_scale, upper_scale = tail['lower_scale'], tail['upper_scale']
    upper_intercept = tail['upper_intercept']

    verts, feasible = group_vertices(
        groups, A_low, b_low, A_high, b_high, tail['l'], tail['u'], x_l, x_u)

    e = group_constants(c_rows, groups, verts, feasible, d)
    if e is None:
        return None

    M, n = c_rows.shape
    flat = groups.reshape(-1)
    grouped = np.zeros(n, dtype=bool)
    grouped[flat] = True
    free = (~grouped)[None, :]

    d_full = np.zeros((M, n))
    d_full[:, flat] = d.reshape(M, -1)

    use_low = c_rows > 0
    pick_low = d_full > 0

    w_low = np.where(free & use_low, c_rows * lower_scale, 0.0)
    w_high = np.where(free & ~use_low, c_rows * upper_scale, 0.0)
    w_low = np.where(grouped[None] & pick_low, d_full, w_low)
    w_high = np.where(grouped[None] & ~pick_low, d_full, w_high)

    total_A = w_low @ A_low + w_high @ A_high
    intercepts = np.where(free & ~use_low, c_rows * upper_intercept, 0.0).sum(axis=1)
    total_b = w_low @ b_low + w_high @ b_high + intercepts + e + const

    return np.maximum(total_A, 0) @ x_l + np.minimum(total_A, 0) @ x_u + total_b


def select_groups(importance, corr_fn, k, max_groups):
    candidates = int((importance > 0).sum())
    if candidates < k:
        return None
    n_cand = min(candidates, k * max_groups)
    cand = np.argsort(-importance)[:n_cand]
    N = cand.size
    if N < k:
        return None

    ii, jj = np.triu_indices(N, k=1)
    if ii.size == 0:
        return None
    scores = corr_fn(cand[ii], cand[jj])
    imp = importance[cand]
    scores = scores * np.sqrt(np.maximum(imp[ii] * imp[jj], 1e-24))

    order = np.argsort(-scores)
    used = np.zeros(N, dtype=bool)
    groups = []
    for pos in order:
        a, b = int(ii[pos]), int(jj[pos])
        if used[a] or used[b]:
            continue
        used[a] = used[b] = True
        groups.append([a, b])
        if len(groups) >= max_groups:
            break
    if not groups:
        return None

    if k >= 3:
        sm = np.zeros((N, N))
        sm[ii, jj] = scores
        sm = sm + sm.T
        extended = []
        for pair in groups:
            free_idx = np.nonzero(~used)[0]
            if free_idx.size == 0:
                break
            combined = sm[pair[0], free_idx] + sm[pair[1], free_idx]
            pick = int(free_idx[int(np.argmax(combined))])
            used[pick] = True
            extended.append(pair + [pick])
        groups = extended
        if not groups:
            return None

    return cand[np.array(groups)]