"""
End-to-end check of the k-ReLU integration on a realistic one-hidden-layer
setting: many neurons, only some of them grouped, the rest keeping their
ordinary DeepPoly lines, everything concretised once over the input box.

This exercises the same code path verifier.py uses for the final margin.

Asserted:
  * refined bound <= true minimum (soundness),
  * refined bound >= plain DeepPoly bound at the DeepPoly-equivalent d,
  * optimising d improves it further and still stays sound.
"""
import numpy as np

import krelu_numpy_shadow as K


def deeppoly_reference(c, tail, x_l, x_u, const):
    """Plain single-neuron DeepPoly bound on c @ ReLU(z) + const."""
    use_low = c > 0
    w_low = np.where(use_low, c * tail['lower_scale'], 0.0)
    w_high = np.where(~use_low, c * tail['upper_scale'], 0.0)
    A = w_low @ tail['A_low'] + w_high @ tail['A_high']
    b = (w_low @ tail['b_low'] + w_high @ tail['b_high']
         + np.where(~use_low, c * tail['upper_intercept'], 0.0).sum() + const)
    return np.maximum(A, 0) @ x_l + np.minimum(A, 0) @ x_u + b


def build_case(seed, n_neurons=14, n_in=4, eps=0.45):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_neurons, n_in)) / np.sqrt(n_in)
    b = rng.normal(size=n_neurons) * 0.4
    x0 = rng.random(n_in)
    x_l = np.clip(x0 - eps, 0, 1)
    x_u = np.clip(x0 + eps, 0, 1)

    A_low = A_high = W
    b_low = b_high = b
    l = np.maximum(W, 0) @ x_l + np.minimum(W, 0) @ x_u + b
    u = np.maximum(W, 0) @ x_u + np.minimum(W, 0) @ x_l + b

    neg = u <= 0
    pos = l >= 0
    cross = ~neg & ~pos
    lam = np.where(u >= -l, 1.0, 0.0)
    lower_scale = np.where(pos, 1.0, np.where(cross, lam, 0.0))
    denom = np.where(cross, np.maximum(u - l, 1e-12), 1.0)
    up_slope = np.where(cross, u / denom, 0.0)
    upper_scale = np.where(pos, 1.0, np.where(cross, up_slope, 0.0))
    upper_intercept = np.where(cross, -up_slope * l, 0.0)

    tail = dict(A_low=A_low, b_low=b_low, A_high=A_high, b_high=b_high,
                l=l, u=u, lower_scale=lower_scale, upper_scale=upper_scale,
                upper_intercept=upper_intercept)
    return rng, W, b, x_l, x_u, tail, cross


def run(seed, k=3, max_groups=3):
    rng, W, b, x_l, x_u, tail, cross = build_case(seed)
    n = W.shape[0]
    if int(cross.sum()) < k * max_groups:
        return None

    c = rng.normal(size=n)
    const = 0.0

    l, u = tail['l'], tail['u']
    lower_gap = np.maximum((1 - tail['lower_scale']) * u,
                           -tail['lower_scale'] * l).clip(min=0)
    err = np.where(c > 0, lower_gap, tail['upper_intercept'])
    importance = np.where(cross, np.abs(c) * err, 0.0)

    def corr_fn(ia, ib):
        s_lo, s_hi, d_lo, d_hi = K.pair_bounds(
            ia, ib, tail['A_low'], tail['b_low'], tail['A_high'],
            tail['b_high'], l, u, x_l, x_u)
        box_sum = (u[ia] + u[ib]) - (l[ia] + l[ib])
        box_diff = (u[ia] - l[ib]) - (l[ia] - u[ib])
        return (box_sum - (s_hi - s_lo)) + (box_diff - (d_hi - d_lo))

    groups = K.select_groups(importance, corr_fn, k, max_groups)
    if groups is None:
        return None

    c_rows = c[None, :]
    d0 = K.deeppoly_d(c_rows, groups, tail['lower_scale'], tail['upper_scale'])

    dp = deeppoly_reference(c, tail, x_l, x_u, const)
    init = K.refine_linear_forms(c_rows, const, groups, d0, tail, x_l, x_u)
    if init is None:
        return None
    init = float(init[0])

    # crude projected ascent on d, mirroring what Adam does in verifier.py
    d = d0.copy()
    best = init
    G, kk = groups.shape
    scale = 0.4 * (np.abs(c).max() + 1e-9)
    for it in range(300):
        eps_fd = 1e-4
        grad = np.zeros_like(d)
        base = K.refine_linear_forms(c_rows, const, groups, d, tail, x_l, x_u)
        if base is None:
            break
        base = float(base[0])
        for g in range(G):
            for j in range(kk):
                dd = d.copy()
                dd[0, g, j] += eps_fd
                v = K.refine_linear_forms(c_rows, const, groups, dd, tail, x_l, x_u)
                grad[0, g, j] = (float(v[0]) - base) / eps_fd
        d = d + (scale / np.sqrt(it + 1)) * grad
        v = K.refine_linear_forms(c_rows, const, groups, d, tail, x_l, x_u)
        if v is not None:
            best = max(best, float(v[0]))

    # ground truth by dense sampling
    X = x_l + (x_u - x_l) * rng.random((250000, x_l.size))
    Z = X @ W.T + b
    truth = float(np.min(np.maximum(Z, 0) @ c))

    return dict(dp=dp, init=init, opt=best, truth=truth,
                n_grouped=int(groups.size), n_unstable=int(cross.sum()))


def main():
    for k, mg in ((3, 3),):
        rows, problems = [], []
        seed = 0
        while len(rows) < 25 and seed < 400:
            r = run(seed, k=k, max_groups=mg)
            seed += 1
            if r is None:
                continue
            rows.append(r)
            if r['init'] > r['truth'] + 1e-6:
                problems.append(f"seed {seed}: init {r['init']:.6f} > truth {r['truth']:.6f}")
            if r['opt'] > r['truth'] + 1e-6:
                problems.append(f"seed {seed}: opt {r['opt']:.6f} > truth {r['truth']:.6f}")
            if r['init'] < r['dp'] - 1e-9:
                problems.append(f"seed {seed}: init {r['init']:.6f} < DeepPoly {r['dp']:.6f}")

        dp = np.array([r['dp'] for r in rows])
        init = np.array([r['init'] for r in rows])
        opt = np.array([r['opt'] for r in rows])
        truth = np.array([r['truth'] for r in rows])
        gap_dp = truth - dp
        gap_opt = truth - opt
        closed = 1 - gap_opt / np.maximum(gap_dp, 1e-9)

        print(f'--- k={k}, {mg} groups, {rows[0]["n_unstable"]}-ish unstable of 14 ---')
        print(f'  neurons jointly relaxed: {rows[0]["n_grouped"]}')
        print(f'  mean gap to truth: DeepPoly {gap_dp.mean():.4f} -> k-ReLU {gap_opt.mean():.4f}')
        print(f'  gap closed: mean {closed.mean():.1%}  median {np.median(closed):.1%}')
        print(f'  improvement at init d (no optimisation): {(init - dp).mean():.4f}')
        print(f'  improvement after optimising d:          {(opt - dp).mean():.4f}')
        if problems:
            print(f'  SOUNDNESS PROBLEMS: {len(problems)}')
            for p in problems[:8]:
                print('    ', p)
        else:
            print('  no violations')
        print()


if __name__ == '__main__':
    main()