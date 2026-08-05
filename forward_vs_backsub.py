"""
Forward symbolic propagation (what backsub_shadow_check.py does) vs true DeepPoly
backsubstitution, on IDENTICAL relaxations and IDENTICAL alpha (0/1 heuristic).

Forward : at each affine layer,  A_low <- W+ @ A_low + W- @ A_high.
          Each neuron's lower/upper line is committed using the sign of the
          LOCAL weight, before the downstream coefficients are known.

Backsub : the target expression is pushed back one layer at a time; the line
          for each neuron is chosen from the sign of the ACCUMULATED
          coefficient, after cancellation along the whole downstream path is
          known.  Every intermediate bound is itself computed this way.

Both are sound; the question is how much precision the early commitment costs.
"""
import numpy as np


def make_net(rng, dims):
    return [(rng.normal(size=(dims[i+1], dims[i]))/np.sqrt(dims[i]),
             rng.normal(size=dims[i+1])*0.1) for i in range(len(dims)-1)]


def fwd(net, x):
    h = x
    for i, (W, b) in enumerate(net):
        h = W @ h + b
        if i+1 < len(net):
            h = np.maximum(h, 0)
    return h


def relax(l, u):
    neg, pos = u <= 0, l >= 0
    cross = ~neg & ~pos
    lam = np.where(u >= -l, 1.0, 0.0)
    lo_s = np.where(pos, 1.0, np.where(cross, lam, 0.0))
    den = np.where(cross, np.maximum(u-l, 1e-12), 1.0)
    sl = np.where(cross, u/den, 0.0)
    up_s = np.where(pos, 1.0, np.where(cross, sl, 0.0))
    up_i = np.where(cross, -sl*l, 0.0)
    return lo_s, up_s, up_i


def clo(A, b, xl, xu): return np.maximum(A, 0) @ xl + np.minimum(A, 0) @ xu + b
def chi(A, b, xl, xu): return np.maximum(A, 0) @ xu + np.minimum(A, 0) @ xl + b


def forward_margin(net, xl, xu, t, g):
    n = xl.size
    Al = np.eye(n); Ah = np.eye(n); bl = np.zeros(n); bh = np.zeros(n)
    unstable = []
    for i, (W, b) in enumerate(net[:-1]):
        Wp, Wn = np.maximum(W, 0), np.minimum(W, 0)
        Al, Ah = Wp@Al + Wn@Ah, Wp@Ah + Wn@Al
        bl, bh = Wp@bl + Wn@bh + b, Wp@bh + Wn@bl + b
        l, u = clo(Al, bl, xl, xu), chi(Ah, bh, xl, xu)
        unstable.append(int(((l < 0) & (u > 0)).sum()))
        lo_s, up_s, up_i = relax(l, u)
        Al = lo_s[:, None]*Al; bl = lo_s*bl
        Ah = up_s[:, None]*Ah; bh = up_s*bh + up_i
    W, b = net[-1]
    c = W[t] - W[g]; d = b[t] - b[g]
    A = np.maximum(c, 0) @ Al + np.minimum(c, 0) @ Ah
    k = np.maximum(c, 0) @ bl + np.minimum(c, 0) @ bh + d
    return clo(A, k, xl, xu), unstable


def backsub_margin(net, xl, xu, t, g):
    """Every intermediate bound obtained by backsubstitution to the input."""
    relaxations = []
    unstable = []

    def push(A, k, upto, lower):
        """Backsubstitute rows A (on post-acts of ReLU layer upto) to the input."""
        for s in range(upto, -1, -1):
            lo_s, up_s, up_i = relaxations[s]
            P, N = np.maximum(A, 0), np.minimum(A, 0)
            if lower:
                k = k + N @ up_i
                A = P*lo_s + N*up_s
            else:
                k = k + P @ up_i
                A = P*up_s + N*lo_s
            Ws, bs = net[s]
            k = k + A @ bs
            A = A @ Ws
        return A, k

    for tt in range(len(net)-1):
        W, b = net[tt]
        if tt == 0:
            A_lo, k_lo, A_hi, k_hi = W.copy(), b.copy(), W.copy(), b.copy()
        else:
            A_lo, k_lo = push(W.copy(), b.copy(), tt-1, True)
            A_hi, k_hi = push(W.copy(), b.copy(), tt-1, False)
        l, u = clo(A_lo, k_lo, xl, xu), chi(A_hi, k_hi, xl, xu)
        unstable.append(int(((l < 0) & (u > 0)).sum()))
        relaxations.append(relax(l, u))

    W, b = net[-1]
    c = (W[t] - W[g])[None, :]
    d = np.array([b[t] - b[g]])
    A, k = push(c, d, len(net)-2, True)
    return float(clo(A, k, xl, xu)[0]), unstable


def trial(seed, dims, eps):
    rng = np.random.default_rng(seed)
    net = make_net(rng, dims)
    x0 = rng.random(dims[0])
    xl, xu = np.clip(x0-eps, 0, 1), np.clip(x0+eps, 0, 1)
    out = fwd(net, x0)
    t = int(np.argmax(out)); g = int(np.argsort(out)[-2])

    fm, uf = forward_margin(net, xl, xu, t, g)
    bm, ub = backsub_margin(net, xl, xu, t, g)

    X = xl + (xu-xl)*rng.random((20000, dims[0]))
    truth = min(float(np.min([fwd(net, x)[t] - fwd(net, x)[g] for x in X[:2000]])), np.inf)
    return float(fm), bm, uf, ub, truth


print(f'{"dims":<24}{"eps":>5}{"forward":>10}{"backsub":>10}{"gain":>9}'
      f'{"sampled":>10}   unstable per layer  fwd -> backsub')
bad = []
for dims in ([20, 30, 30, 10], [20, 40, 40, 40, 10], [20, 50, 50, 50, 50, 10]):
    for eps in (0.05, 0.10, 0.20):
        F, B, T = [], [], []
        uf = ub = None
        for s in range(10):
            fm, bm, a, b, tr = trial(s, dims, eps)
            F.append(fm); B.append(bm); T.append(tr)
            if fm > tr + 1e-7: bad.append(('forward unsound', dims, eps, s))
            if bm > tr + 1e-7: bad.append(('backsub unsound', dims, eps, s))
            if s == 0: uf, ub = a, b
        print(f'{str(dims):<24}{eps:>5.2f}{np.mean(F):>10.4f}{np.mean(B):>10.4f}'
              f'{np.mean(B)-np.mean(F):>9.4f}{np.mean(T):>10.4f}   {uf} -> {ub}')
print()
print('soundness violations:', bad if bad else 'none')