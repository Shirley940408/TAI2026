"""
NumPy shadow of _Verifier.backsub (fc path: Normalization / Flatten / Linear /
ReLU), transcribed from verifier.py, checked against the independent
reference implementation in bvf.py and against sampling.
"""
import numpy as np
import forward_vs_backsub as bvf


def backsub(A, k, i0, lower, layers, relaxations, shapes_in):
    """Mirror of verifier.py::_Verifier.backsub."""
    for i in range(i0, -1, -1):
        kind, payload = layers[i]

        if kind == 'relu':
            lower_scale, upper_scale, upper_intercept = relaxations[i]
            P, N = np.maximum(A, 0), np.minimum(A, 0)
            if lower:
                k = k + N @ upper_intercept
                A = P * lower_scale + N * upper_scale
            else:
                k = k + P @ upper_intercept
                A = P * upper_scale + N * lower_scale
            continue

        if kind == 'flatten':
            continue

        if kind == 'linear':
            W, b = payload
            if b is not None:
                k = k + A @ b
            A = A @ W
            continue

        if kind == 'norm':
            mean, sigma = payload
            k = k + A @ (-mean / sigma)
            A = A / sigma
            continue

        raise NotImplementedError(kind)
    return A, k


def clo(A, b, xl, xu): return np.maximum(A, 0) @ xl + np.minimum(A, 0) @ xu + b
def chi(A, b, xl, xu): return np.maximum(A, 0) @ xu + np.minimum(A, 0) @ xl + b


def run(seed, dims, eps, with_norm=True):
    rng = np.random.default_rng(seed)
    net = bvf.make_net(rng, dims)
    x0 = rng.random(dims[0])
    xl, xu = np.clip(x0 - eps, 0, 1), np.clip(x0 + eps, 0, 1)

    mean = np.full(dims[0], 0.1307) if with_norm else np.zeros(dims[0])
    sigma = np.full(dims[0], 0.3081) if with_norm else np.ones(dims[0])

    # body layers as the verifier sees them: norm, flatten, [linear, relu]*
    layers = [('norm', (mean, sigma)), ('flatten', None)]
    for W, b in net[:-1]:
        layers.append(('linear', (W, b)))
        layers.append(('relu', None))

    def forward(x):
        h = (x - mean) / sigma
        for i, (W, b) in enumerate(net):
            h = W @ h + b
            if i + 1 < len(net):
                h = np.maximum(h, 0)
        return h

    relaxations = [None] * len(layers)
    for i, (kind, _) in enumerate(layers):
        if kind != 'relu':
            continue
        n = net[(i - 3) // 2][0].shape[0]
        eye = np.eye(n)
        zero = np.zeros(n)
        A_lo, k_lo = backsub(eye, zero, i - 1, True, layers, relaxations, None)
        A_hi, k_hi = backsub(eye, zero, i - 1, False, layers, relaxations, None)
        l, u = clo(A_lo, k_lo, xl, xu), chi(A_hi, k_hi, xl, xu)
        relaxations[i] = bvf.relax(l, u)

    out = forward(x0)
    t, g = int(np.argmax(out)), int(np.argsort(out)[-2])
    W, b = net[-1]
    c = (W[t] - W[g])[None, :]
    d = np.array([b[t] - b[g]])
    A, k = backsub(c, d, len(layers) - 1, True, layers, relaxations, None)
    margin = float(clo(A, k, xl, xu)[0])

    X = xl + (xu - xl) * rng.random((4000, dims[0]))
    truth = float(min(forward(x)[t] - forward(x)[g] for x in X))

    # reference: bvf on the normalised network (fold normalisation into W1)
    net2 = [((net[0][0] / sigma), net[0][1] - net[0][0] @ (mean / sigma))] + net[1:]
    ref, _ = bvf.backsub_margin(net2, xl, xu, t, g)
    return margin, ref, truth


bad = []
print(f'{"dims":<22}{"eps":>6}{"shadow":>11}{"reference":>11}{"sampled":>11}')
for dims in ([12, 20, 20, 6], [12, 25, 25, 25, 6], [12, 30, 30, 30, 30, 6]):
    for eps in (0.05, 0.15):
        ms, rs, ts = [], [], []
        for s in range(8):
            m, r, t = run(s, dims, eps)
            ms.append(m); rs.append(r); ts.append(t)
            if m > t + 1e-7:
                bad.append(('unsound', dims, eps, s, m, t))
            if abs(m - r) > 1e-8:
                bad.append(('mismatch vs reference', dims, eps, s, m, r))
        print(f'{str(dims):<22}{eps:>6.2f}{np.mean(ms):>11.4f}'
              f'{np.mean(rs):>11.4f}{np.mean(ts):>11.4f}')
print()
print('problems:', bad if bad else 'none  (shadow == independent reference, and always below sampled truth)')