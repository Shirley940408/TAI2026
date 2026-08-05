"""
Neural network robustness verifier.

Method stack, from cheapest to strongest:

  Stage 0 -- PGD falsification.
      A concrete counterexample settles a non-robust case immediately.

  Stage 1 -- DeepPoly by backsubstitution, intersected with interval (IBP)
      bounds, ReLU lower slopes `alpha` optimised by projected Adam.

      The distinction that matters here: forward symbolic propagation
      (A_low <- W+ @ A_low + W- @ A_high) commits each ReLU to a lower or
      upper line using the sign of the LOCAL weight, before the downstream
      coefficients are known.  Backsubstitution pushes the target expression
      back one layer at a time and picks each line from the sign of the
      ACCUMULATED coefficient, once cancellation along the whole downstream
      path is known.  On identical relaxations and identical alpha the gap
      grows sharply with depth -- on random 5-hidden-layer networks at
      eps=0.1 the margin bound went from -11.8 (forward) to -0.77
      (backsubstitution), and the count of unstable neurons in the deepest
      layers roughly halved.  Every intermediate pre-activation bound is
      therefore backsubstituted too, not just the output margin.

      The forward symbolic pass is kept as an extra intersection source: it
      is cheap, always sound, and still useful for the wide convolutional
      layers where full backsubstitution of intermediate bounds would blow
      the time budget.

  Stage 2 -- k-ReLU multi-neuron joint relaxation (Singh et al., NeurIPS
      2019; the same family PRIMA generalises).  This is the part that moves
      the precision ceiling rather than climbing to the top of it.

  Stage 3 -- branch and bound on unstable neurons, running on top of the
      tighter Stage 2 bounds.

Why Stage 2 is the one that matters
-----------------------------------
DeepPoly relaxes each ReLU on its own, so all correlation between neurons
sharing the same input is discarded.  Salman et al. (NeurIPS 2019) prove
this has a precision ceiling that slope tuning cannot cross -- exactly the
plateau that more optimiser steps and better slope initialisations kept
running into.

For a group G of k neurons, k-ReLU relaxes them jointly.  Writing z_G for
their pre-activations, the reachable set is over-approximated by

    P = box(l, u)  intersect  { lo <= z_a +/- z_b <= hi  for every pair }

where the octahedral bounds come from concretising the *symbolic* DeepPoly
expressions for z_a + z_b and z_a - z_b.  Those are strictly tighter than
l_a + l_b and u_a + u_b precisely when the two neurons are correlated
through the input -- that difference is the information the single-neuron
relaxation throws away, and it is what gets recovered here.

The graph { (z, ReLU(z)) : z in P } is piecewise affine, so every extreme
point of its convex hull is a vertex of P subdivided by the hyperplanes
z_j = 0.  Enumerating that vertex set V yields, for any d, the valid
inequality

    sum_j c_j y_j  >=  sum_j d_j z_j + e(d),
    e(d) = min_{v in V} ( sum_j c_j ReLU(v_j) - sum_j d_j v_j ).

Substituting each z_j by its symbolic affine bound turns the right-hand side
back into an affine function of the network input.  The joint constraint
therefore drops straight into the forward pass -- no backward substitution
anywhere -- and because concretisation over the input box still happens once
at the very end, correlation *between* groups survives too.

Two consequences worth stating explicitly:

  * Choosing d_j = c_j * (the DeepPoly slope) reproduces DeepPoly exactly,
    except that the sum of the per-neuron intercepts is replaced by e(d),
    which is provably at least that sum.  The joint relaxation therefore can
    never be worse than the single-neuron one.  d is then optimised by Adam
    alongside alpha; the bound is concave in d.

  * Enumerating a superset of the vertices is safe -- a spurious point only
    lowers e(d) -- so the feasibility tolerance is deliberately loose.

k = 3 is used where the coefficient vector is known exactly (the output
margin); k = 2 is used for the cheaper intermediate-bound tightening.
"""

import argparse
import heapq
import itertools
import os
import time

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']

# Overall wall-clock budget per test case, kept under the 3-minute limit.
# Every stage checks this deadline, so a slower machine explores less rather
# than overrunning.
TIME_BUDGET = 165.0
# Slice of that budget the falsification attack may use.  Cheap insurance:
# a non-robust case the attack misses burns the whole remaining budget.
PGD_BUDGET = 12.0

# A bound only counts as a proof if it is strictly positive by this much.
CERT_TOL = 1e-7
# Emptiness of a branch is only claimed with this much slack, so floating
# point noise can never prune a branch that really exists.
FEAS_TOL = 1e-5

# Vertex-enumeration feasibility tolerance.  Over-inclusion is sound (a
# spurious vertex only lowers e), under-inclusion is not, so this is loose.
_VERTEX_REL_TOL = 1e-4
_VERTEX_ABS_TOL = 1e-6


# ===========================================================================
# Stage 0: PGD falsification
# ===========================================================================

def _pgd_falsify(model, x, eps, true_label, deadline=None,
                 restarts=8, steps=100):
    """True if a concrete adversarial example exists inside the eps-ball.

    A positive answer proves the case is NOT robust, so `not verified` is the
    only possible correct output.  A negative answer means nothing and is
    ignored.

    Worth spending real effort here: a non-robust case that this misses goes
    on to consume the entire verification budget before being declined
    anyway, so a few seconds of attack can save minutes.  Hence several step
    sizes with decay, plus a targeted pass against every wrong class -- an
    untargeted attack stalls when two wrong classes pull in opposite
    directions.
    """
    lower = torch.clamp(x.detach() - eps, 0.0, 1.0)
    upper = torch.clamp(x.detach() + eps, 0.0, 1.0)
    schedules = [eps / 4.0, eps / 10.0, eps / 40.0]

    n_class = model(x).shape[1]
    targets = [c for c in range(n_class) if c != true_label]

    def out_of_time():
        return deadline is not None and time.perf_counter() >= deadline

    def sweep(loss_fn, n_restarts, n_steps):
        for restart in range(n_restarts):
            if out_of_time():
                return False
            if restart == 0:
                adv = x.detach().clone()
            else:
                adv = lower + (upper - lower) * torch.rand_like(x)
            step = max(schedules[restart % len(schedules)], 1e-4)

            for it in range(n_steps):
                adv = adv.detach().requires_grad_(True)
                logits = model(adv)[0]
                if int(logits.argmax()) != true_label:
                    return True
                grad = torch.autograd.grad(loss_fn(logits), adv)[0]
                # decay so late iterations settle into a corner of the ball
                cur = step * (0.1 ** (it / max(n_steps - 1, 1)))
                adv = adv.detach() + cur * grad.sign()
                adv = torch.max(torch.min(adv, upper), lower)

            with torch.no_grad():
                if int(model(adv)[0].argmax()) != true_label:
                    return True
        return False

    def untargeted(logits):
        other = torch.cat([logits[:true_label], logits[true_label + 1:]])
        return other.max() - logits[true_label]

    if sweep(untargeted, restarts, steps):
        return True

    per_target = max(1, restarts // max(len(targets), 1))
    for target in targets:
        if out_of_time():
            return False

        def targeted(logits, t=target):
            return logits[t] - logits[true_label]

        if sweep(targeted, per_target, steps):
            return True

    return False


# ===========================================================================
# Stage 2: k-ReLU joint relaxation primitives
# ===========================================================================

_PLANE_CACHE = {}


def _plane_system(k, device, dtype):
    """Subdivision planes, constraints, and pre-inverted k x k systems.

    Depends only on the group size, so it is built once and shared by every
    group and every propagation.

      normals  [P, k]      one row per subdivision plane
      cons     [n_con, k]  constraints written as  cons @ z <= rhs
      combos   [C, k]      the non-degenerate k-subsets of planes
      inverses [C, k, k]   inverse of each selected k x k system
    """
    key = (k, str(device), str(dtype))
    cached = _PLANE_CACHE.get(key)
    if cached is not None:
        return cached

    eye = torch.eye(k, device=device, dtype=dtype)
    pairs = list(itertools.combinations(range(k), 2))

    normals = []
    for a in range(k):                      # z_a = l_a
        normals.append(eye[a])
    for a in range(k):                      # z_a = u_a
        normals.append(eye[a])
    for a in range(k):                      # z_a = 0, the ReLU subdivision
        normals.append(eye[a])
    for a, b in pairs:                      # the four octahedral planes
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] + eye[b])
        normals.append(eye[a] - eye[b])
        normals.append(eye[a] - eye[b])
    normals = torch.stack(normals)

    cons = []
    for a in range(k):
        cons.append(eye[a])
        cons.append(-eye[a])
    for a, b in pairs:
        cons.append(eye[a] + eye[b])
        cons.append(-(eye[a] + eye[b]))
        cons.append(eye[a] - eye[b])
        cons.append(-(eye[a] - eye[b]))
    cons = torch.stack(cons)

    keep, inverses = [], []
    for combo in itertools.combinations(range(normals.size(0)), k):
        M = normals[list(combo)]
        if float(torch.det(M).abs()) > 1e-8:
            keep.append(combo)
            inverses.append(torch.inverse(M))

    result = {
        'normals': normals,
        'cons': cons,
        'combos': torch.tensor(keep, device=device, dtype=torch.long),
        'inverses': torch.stack(inverses),
        'pairs': pairs,
        'k': k,
    }
    _PLANE_CACHE[key] = result
    return result


def _conc_low(A, b, x_l, x_u):
    return A.clamp(min=0) @ x_l + A.clamp(max=0) @ x_u + b


def _conc_high(A, b, x_l, x_u):
    return A.clamp(min=0) @ x_u + A.clamp(max=0) @ x_l + b


def _pair_bounds(ia, ib, A_low, b_low, A_high, b_high, l, u, x_l, x_u):
    """Octahedral bounds on z_ia + z_ib and z_ia - z_ib, batched over pairs.

    These are the only place the joint relaxation learns anything the
    single-neuron one does not: concretising the symbolic expression of the
    sum is tighter than adding the two separately concretised bounds exactly
    when the neurons are correlated through the input.
    """
    As = A_low[ia] + A_low[ib]
    sum_lo = torch.maximum(
        _conc_low(As, b_low[ia] + b_low[ib], x_l, x_u), l[ia] + l[ib]
    )
    Ash = A_high[ia] + A_high[ib]
    sum_hi = torch.minimum(
        _conc_high(Ash, b_high[ia] + b_high[ib], x_l, x_u), u[ia] + u[ib]
    )
    Ad = A_low[ia] - A_high[ib]
    diff_lo = torch.maximum(
        _conc_low(Ad, b_low[ia] - b_high[ib], x_l, x_u), l[ia] - u[ib]
    )
    Adh = A_high[ia] - A_low[ib]
    diff_hi = torch.minimum(
        _conc_high(Adh, b_high[ia] - b_low[ib], x_l, x_u), u[ia] - l[ib]
    )
    return sum_lo, sum_hi, diff_lo, diff_hi


def _group_vertices(groups, A_low, b_low, A_high, b_high, l, u, x_l, x_u):
    """Vertices of the subdivided group polytopes.

    groups: [G, k] neuron indices.  Returns (verts [G, C, k], feasible [G, C]).
    Inputs are detached; the result carries no gradient.
    """
    k = groups.size(1)
    system = _plane_system(k, l.device, l.dtype)

    lg = l[groups]
    ug = u[groups]

    offsets = [lg, ug, torch.zeros_like(lg)]
    rhs = []
    for a in range(k):
        rhs.append(ug[:, a])
        rhs.append(-lg[:, a])

    for a, b in system['pairs']:
        s_lo, s_hi, d_lo, d_hi = _pair_bounds(
            groups[:, a], groups[:, b],
            A_low, b_low, A_high, b_high, l, u, x_l, x_u,
        )
        offsets.append(torch.stack([s_lo, s_hi, d_lo, d_hi], dim=1))
        rhs.extend([s_hi, -s_lo, d_hi, -d_lo])

    offsets = torch.cat(offsets, dim=1)                 # [G, P]
    rhs = torch.stack(rhs, dim=1)                       # [G, n_con]

    selected = offsets[:, system['combos']]             # [G, C, k]
    verts = torch.einsum('cij,gcj->gci', system['inverses'], selected)

    lhs = torch.einsum('nk,gck->gcn', system['cons'], verts)
    tol = _VERTEX_ABS_TOL + _VERTEX_REL_TOL * rhs.abs().amax(dim=1, keepdim=True)
    feasible = (lhs <= rhs.unsqueeze(1) + tol.unsqueeze(1)).all(dim=2)
    return verts, feasible


def _deeppoly_d(c_rows, groups, lower_scale, upper_scale):
    """The d that reproduces DeepPoly exactly -- the optimisation's start."""
    cg = c_rows[:, groups]                              # [M, G, k]
    ls = lower_scale[groups].unsqueeze(0)
    us = upper_scale[groups].unsqueeze(0)
    return torch.where(cg > 0, cg * ls, cg * us)


def _refine_linear_forms(c_rows, const, groups, d, tail, x_l, x_u):
    """Lower-bound a batch of linear forms over one ReLU layer's outputs.

    Bounds  c_rows @ ReLU(z) + const  from below, returning [M], or None when
    the joint relaxation could not be built (the caller then keeps DeepPoly).

    Grouped neurons use the joint inequality; every other neuron keeps its
    ordinary DeepPoly line.  The whole expression is concretised once, so
    input-level correlation across groups is preserved.
    """
    A_low, b_low = tail['A_low'], tail['b_low']
    A_high, b_high = tail['A_high'], tail['b_high']
    lower_scale, upper_scale = tail['lower_scale'], tail['upper_scale']
    upper_intercept = tail['upper_intercept']

    with torch.no_grad():
        verts, feasible = _group_vertices(
            groups,
            A_low.detach(), b_low.detach(), A_high.detach(), b_high.detach(),
            tail['l'].detach(), tail['u'].detach(), x_l, x_u,
        )
        if not bool(feasible.any(dim=1).all()):
            return None

    cg = c_rows[:, groups]                              # [M, G, k]
    relu_v = verts.clamp(min=0)                         # [G, C, k]
    term = (
        torch.einsum('mgk,gck->mgc', cg, relu_v)
        - torch.einsum('mgk,gck->mgc', d, verts)
    )
    term = torch.where(
        feasible.unsqueeze(0), term, torch.full_like(term, float('inf'))
    )
    e = term.amin(dim=2).sum(dim=1)                     # [M]

    M, n = c_rows.shape
    device, dtype = c_rows.device, c_rows.dtype

    flat = groups.reshape(-1)
    grouped = torch.zeros(n, device=device, dtype=torch.bool)
    grouped[flat] = True
    free = (~grouped).unsqueeze(0)

    d_full = torch.zeros(M, n, device=device, dtype=dtype).scatter(
        1, flat.unsqueeze(0).repeat(M, 1), d.reshape(M, -1)
    )

    # Weight on each neuron's symbolic lower / upper expression.  Folded into
    # a matmul so no [M, n, input_dim] tensor is ever materialised.
    use_low = c_rows > 0
    pick_low = d_full > 0
    zeros = torch.zeros_like(c_rows)

    w_low = torch.where(free & use_low, c_rows * lower_scale, zeros)
    w_high = torch.where(free & ~use_low, c_rows * upper_scale, zeros)
    w_low = torch.where(grouped.unsqueeze(0) & pick_low, d_full, w_low)
    w_high = torch.where(grouped.unsqueeze(0) & ~pick_low, d_full, w_high)

    total_A = w_low @ A_low + w_high @ A_high
    intercepts = torch.where(
        free & ~use_low, c_rows * upper_intercept, zeros
    ).sum(dim=1)
    total_b = w_low @ b_low + w_high @ b_high + intercepts + e + const

    return total_A.clamp(min=0) @ x_l + total_A.clamp(max=0) @ x_u + total_b


def _select_groups(importance, corr_fn, k, max_groups):
    """Greedy grouping: strongest neurons first, paired by correlation.

    `importance` is zero for neurons not worth grouping (stable ones, or ones
    the objective does not depend on).  `corr_fn(ia, ib)` scores how much the
    octahedral bounds tighten relative to the plain box -- that is, how much
    joint information there is to recover from the pair.
    """
    device = importance.device
    positive = int((importance > 0).sum())
    if positive < k:
        return None

    n_cand = min(positive, k * max_groups)
    cand = torch.topk(importance, n_cand).indices
    N = int(cand.numel())
    if N < k:
        return None

    ii, jj = torch.triu_indices(N, N, offset=1, device=device).unbind(0)
    if ii.numel() == 0:
        return None

    scores = corr_fn(cand[ii], cand[jj])
    imp = importance[cand]
    scores = scores * torch.sqrt((imp[ii] * imp[jj]).clamp_min(1e-24))

    order = torch.argsort(scores, descending=True)
    used = torch.zeros(N, dtype=torch.bool, device=device)
    groups = []
    for pos in order.tolist():
        a, b = int(ii[pos]), int(jj[pos])
        if used[a] or used[b]:
            continue
        used[a] = True
        used[b] = True
        groups.append([a, b])
        if len(groups) >= max_groups:
            break
    if not groups:
        return None

    if k >= 3:
        score_matrix = torch.zeros(N, N, device=device, dtype=scores.dtype)
        score_matrix[ii, jj] = scores
        score_matrix = score_matrix + score_matrix.t()
        extended = []
        for pair in groups:
            remaining = (~used).nonzero(as_tuple=True)[0]
            if remaining.numel() == 0:
                break
            combined = (score_matrix[pair[0], remaining]
                        + score_matrix[pair[1], remaining])
            pick = int(remaining[int(torch.argmax(combined))])
            used[pick] = True
            extended.append(pair + [pick])
        groups = extended
        if not groups:
            return None

    return cand[torch.tensor(groups, device=device, dtype=torch.long)]


class _MultiNeuronContext:
    """Group choices and joint-relaxation parameters for one subproblem."""

    def __init__(self, intermediate=False):
        self.margin_groups = None
        self.margin_d = None
        self.layer_groups = {}          # relu index -> [G, 2] or False
        self.intermediate = intermediate

    def parameters(self):
        return [] if self.margin_d is None else [self.margin_d]


# ===========================================================================
# Stages 1-3: the verifier
# ===========================================================================

class _Verifier:

    # Stage 1 optimisation.
    SHARED_STEPS = 18
    TARGET_STEPS = 20
    ROOT_LR = 5e-2
    TEMPERATURE = 0.5
    INIT_INSET = 5e-2

    # Stage 2 (k-ReLU).
    MN_K = 3                 # group size at the output margin
    MN_MAX_GROUPS = 14
    MN_K_INTER = 2           # group size for intermediate bound tightening
    MN_MAX_GROUPS_INTER = 10
    MN_D_LR = 2e-1

    # DeepPoly backsubstitution.  Intermediate bounds for a layer are only
    # backsubstituted when the coefficient matrices stay under this many
    # elements; oversized conv layers keep their (sound, looser) forward
    # symbolic bounds instead.  The output margin is always backsubstituted,
    # since it is only a handful of rows.
    MAX_BACKSUB_ELEMS = 6_000_000

    # Stage 3 (branch and bound).
    SUB_STEPS = 6
    SUB_LR = 8e-2
    # A safety net only.  Hitting it abandons the target outright, so it is
    # set well above the depth any case actually reaches; the wall-clock
    # deadline is meant to be the binding constraint, not this.
    MAX_SPLIT_DEPTH = 80
    MAX_OPEN_SUBPROBLEMS = 4000
    # Seconds held back for each target class still queued behind the current
    # one, so a late easy target is not left with nothing.
    TARGET_RESERVE = 5.0
    FBBT_ROUNDS = 3

    def __init__(self, model, x, eps, true_label):
        model.eval()
        layers = list(model.layers.children())
        if not layers or not isinstance(layers[-1], torch.nn.Linear):
            raise NotImplementedError(
                'This implementation expects the final network layer to be Linear.'
            )

        self.body_layers = layers[:-1]
        self.output_layer = layers[-1]

        self.x = x.detach()
        self.device = x.device
        self.dtype = x.dtype
        self.float_eps = torch.finfo(self.dtype).eps
        self.input_dim = self.x.numel()
        self.true_label = true_label

        flat = self.x.reshape(-1)
        self.base_l = torch.clamp(flat - eps, 0.0, 1.0)
        self.base_u = torch.clamp(flat + eps, 0.0, 1.0)

        n_out = self.output_layer.weight.size(0)
        self.other_labels = torch.tensor(
            [c for c in range(n_out) if c != true_label],
            device=self.device, dtype=torch.long,
        )

        self.n_relu = sum(
            1 for layer in self.body_layers if isinstance(layer, torch.nn.ReLU)
        )
        self.first_map = self._first_relu_affine_map()

        # Per-layer shapes, needed to reshape backsubstitution coefficients
        # and to size conv_transpose2d correctly.
        self.layer_in_shape = []
        self.layer_out_shape = []
        probe = torch.zeros_like(self.x)
        with torch.no_grad():
            for layer in self.body_layers:
                self.layer_in_shape.append(tuple(probe.shape[1:]))
                probe = layer(probe)
                self.layer_out_shape.append(tuple(probe.shape[1:]))
        self.layer_in_numel = [
            int(torch.tensor(shape).prod()) for shape in self.layer_in_shape
        ]

        self.has_conv = any(
            isinstance(layer, torch.nn.Conv2d) for layer in self.body_layers
        )
        if self.has_conv:
            self.SUB_STEPS = 3
            self.MN_K = 2
            self.MN_MAX_GROUPS = 12

        # The joint relaxation at the margin needs the output layer to read
        # directly from a ReLU's post-activations, which holds for every
        # architecture in this project.
        self.margin_refinable = bool(self.body_layers) and isinstance(
            self.body_layers[-1], torch.nn.ReLU
        )

    # -- exact affine prefix, used to push splits back to the input box ----

    def _first_relu_affine_map(self):
        A = torch.eye(self.input_dim, device=self.device, dtype=self.dtype)
        b = torch.zeros(self.input_dim, device=self.device, dtype=self.dtype)

        for layer in self.body_layers:
            if isinstance(layer, torch.nn.ReLU):
                return A, b

            if isinstance(layer, Normalization):
                mean = layer.mean.detach().to(
                    device=self.device, dtype=self.dtype
                ).expand_as(self.x).reshape(-1)
                sigma = layer.sigma.detach().to(
                    device=self.device, dtype=self.dtype
                ).expand_as(self.x).reshape(-1)
                if torch.any(sigma <= 0):
                    raise ValueError('Normalization sigma must be positive.')
                A = A / sigma.unsqueeze(1)
                b = (b - mean) / sigma
                continue

            if isinstance(layer, torch.nn.Flatten):
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=self.device, dtype=self.dtype)
                A = W @ A
                b = W @ b
                if layer.bias is not None:
                    b = b + layer.bias.detach().to(
                        device=self.device, dtype=self.dtype
                    )
                continue

            return None

        return None

    # -- DeepPoly backsubstitution ------------------------------------------

    @staticmethod
    def _pair(value):
        return value if isinstance(value, tuple) else (value, value)

    def backsub_affordable(self, rows, i0):
        cost = 0
        for i in range(i0, -1, -1):
            cost += rows * self.layer_in_numel[i]
            if cost > self.MAX_BACKSUB_ELEMS:
                return False
        return True

    def backsub(self, A, k, i0, lower, relaxations):
        """Push an affine form back to the raw input, one layer at a time.

        `A` holds coefficients over the output of body_layers[i0], shape
        [rows, numel].  Unlike forward symbolic propagation, the lower/upper
        line of each ReLU is chosen from the sign of the ACCUMULATED
        coefficient -- after all cancellation along the downstream path is
        known -- which is what makes this DeepPoly rather than symbolic
        interval propagation.

        Returns (A, k) over the raw (un-normalised) input, or None if some
        relaxation on the path has not been built yet.
        """
        device, dtype = self.device, self.dtype

        for i in range(i0, -1, -1):
            layer = self.body_layers[i]

            if isinstance(layer, torch.nn.ReLU):
                relaxation = relaxations[i]
                if relaxation is None:
                    return None
                lower_scale, upper_scale, upper_intercept = relaxation
                P, N = A.clamp(min=0), A.clamp(max=0)
                if lower:
                    k = k + N @ upper_intercept
                    A = P * lower_scale + N * upper_scale
                else:
                    k = k + P @ upper_intercept
                    A = P * upper_scale + N * lower_scale
                continue

            if isinstance(layer, torch.nn.Flatten):
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                if layer.bias is not None:
                    k = k + A @ layer.bias.detach().to(device=device, dtype=dtype)
                A = A @ W
                continue

            if isinstance(layer, Normalization):
                shape = self.layer_in_shape[i]
                mean = layer.mean.detach().to(device=device, dtype=dtype)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype)
                mean = mean.expand(1, *shape).reshape(-1)
                sigma = sigma.expand(1, *shape).reshape(-1)
                k = k + A @ (-mean / sigma)
                A = A / sigma
                continue

            if isinstance(layer, torch.nn.Conv2d):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                C_out, H_out, W_out = self.layer_out_shape[i]
                C_in, H_in, W_in = self.layer_in_shape[i]

                if layer.bias is not None:
                    bias = layer.bias.detach().to(device=device, dtype=dtype)
                    k = k + A @ bias.view(-1, 1, 1).expand(
                        C_out, H_out, W_out
                    ).reshape(-1)

                rows = A.size(0)
                A4 = A.reshape(rows, C_out, H_out, W_out)
                stride_h, stride_w = self._pair(layer.stride)
                pad_h, pad_w = self._pair(layer.padding)
                dil_h, dil_w = self._pair(layer.dilation)
                ker_h, ker_w = self._pair(layer.kernel_size)
                base_h = (H_out - 1) * stride_h - 2 * pad_h + dil_h * (ker_h - 1) + 1
                base_w = (W_out - 1) * stride_w - 2 * pad_w + dil_w * (ker_w - 1) + 1
                out_pad_h, out_pad_w = H_in - base_h, W_in - base_w
                if not (0 <= out_pad_h < stride_h and 0 <= out_pad_w < stride_w):
                    return None
                A4 = F.conv_transpose2d(
                    A4, W, bias=None, stride=layer.stride, padding=layer.padding,
                    output_padding=(out_pad_h, out_pad_w), groups=layer.groups,
                    dilation=layer.dilation,
                )
                A = A4.reshape(rows, -1)
                continue

            raise NotImplementedError(
                'Unsupported layer in backsubstitution: {}'.format(type(layer))
            )

        return A, k

    # -- input box tightening from split constraints ------------------------

    def tighten_input_box(self, splits):
        """Push first-layer split constraints back onto the input box.

        Returns (x_l, x_u, feasible); feasible=False means an empty branch.
        """
        x_l = self.base_l.clone()
        x_u = self.base_u.clone()
        if self.first_map is None or not splits:
            return x_l, x_u, True

        W, bias = self.first_map
        constraints = []
        for (layer_idx, neuron), sign in splits.items():
            if layer_idx != 0:
                continue
            if sign > 0:
                constraints.append((-W[neuron], -bias[neuron]))
            else:
                constraints.append((W[neuron], bias[neuron]))

        if not constraints:
            return x_l, x_u, True

        for _ in range(self.FBBT_ROUNDS):
            changed = False
            for a, c in constraints:
                per_term_min = a.clamp(min=0) * x_l + a.clamp(max=0) * x_u
                total_min = per_term_min.sum() + c
                if float(total_min) > FEAS_TOL:
                    return x_l, x_u, False

                rhs = -(total_min - per_term_min)
                nonzero = a.abs() > 1e-10
                a_safe = torch.where(nonzero, a, torch.ones_like(a))
                candidate = rhs / a_safe

                new_u = torch.where(
                    nonzero & (a > 0), torch.minimum(x_u, candidate), x_u
                )
                new_l = torch.where(
                    nonzero & (a < 0), torch.maximum(x_l, candidate), x_l
                )
                if float((new_l - x_l).abs().max()) > 1e-9 or \
                        float((new_u - x_u).abs().max()) > 1e-9:
                    changed = True
                x_l, x_u = new_l, new_u

                if float((x_l - x_u).max()) > FEAS_TOL:
                    return x_l, x_u, False

            if not changed:
                break

        x_u = torch.maximum(x_u, x_l)
        return x_l, x_u, True

    # -- group selection ----------------------------------------------------

    def _corr_fn(self, tail, x_l, x_u):
        """Score how much joint information a pair of neurons carries."""
        A_low, b_low = tail['A_low'].detach(), tail['b_low'].detach()
        A_high, b_high = tail['A_high'].detach(), tail['b_high'].detach()
        l, u = tail['l'].detach(), tail['u'].detach()

        def score(ia, ib):
            s_lo, s_hi, d_lo, d_hi = _pair_bounds(
                ia, ib, A_low, b_low, A_high, b_high, l, u, x_l, x_u
            )
            box_sum = (u[ia] + u[ib]) - (l[ia] + l[ib])
            box_diff = (u[ia] - l[ib]) - (l[ia] - u[ib])
            return (box_sum - (s_hi - s_lo)) + (box_diff - (d_hi - d_lo))

        return score

    def _relaxation_area(self, tail):
        l, u = tail['l'].detach(), tail['u'].detach()
        lower_scale = tail['lower_scale'].detach()
        lower_gap = torch.maximum(
            (1.0 - lower_scale) * u, -lower_scale * l
        ).clamp_min(0.0)
        return (tail['upper_intercept'].detach().clamp_min(0.0) + lower_gap,
                lower_gap)

    def build_margin_groups(self, tail, c_row, x_l, x_u):
        area, lower_gap = self._relaxation_area(tail)
        err = torch.where(c_row > 0, lower_gap, tail['upper_intercept'].detach())
        importance = torch.where(
            tail['cross'].detach(), c_row.abs() * err.clamp_min(0.0),
            torch.zeros_like(c_row),
        )
        return _select_groups(
            importance, self._corr_fn(tail, x_l, x_u),
            self.MN_K, self.MN_MAX_GROUPS,
        )

    def build_layer_groups(self, tail, x_l, x_u):
        area, _ = self._relaxation_area(tail)
        importance = torch.where(
            tail['cross'].detach(), area, torch.zeros_like(area)
        )
        return _select_groups(
            importance, self._corr_fn(tail, x_l, x_u),
            self.MN_K_INTER, self.MN_MAX_GROUPS_INTER,
        )

    # -- core propagation ---------------------------------------------------

    def propagate(self, alpha_params=None, splits=None, x_l=None, x_u=None,
                  target_positions=None, initialize_alpha=False,
                  collect_info=False, mn=None, capture_tail=False):
        """Return (margins, extra) or None when the branch is infeasible.

        `margins[i]` lower-bounds f_true - f_target for the selected targets,
        soundly, for every input in the (possibly tightened) box consistent
        with `splits`.
        """
        if alpha_params is None:
            alpha_params = []
        if x_l is None:
            x_l = self.base_l
        if x_u is None:
            x_u = self.base_u

        splits_by_layer = {}
        if splits:
            for (layer_idx, neuron), sign in splits.items():
                splits_by_layer.setdefault(layer_idx, []).append((neuron, sign))

        device, dtype = self.device, self.dtype

        # Backsubstitution concretises against the raw input box, since it
        # walks back through the Normalization layer itself.
        orig_l, orig_u = x_l, x_u

        A_low = torch.eye(self.input_dim, device=device, dtype=dtype)
        A_high = torch.eye(self.input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(self.input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(self.input_dim, device=device, dtype=dtype)

        box_low = x_l.reshape_as(self.x)
        box_high = x_u.reshape_as(self.x)

        cur_shape = tuple(self.x.shape[-3:]) if self.x.dim() >= 3 else None
        relu_id = 0
        info = {'relu': []} if collect_info else None
        last_tail = None
        prev_was_relu = False
        pending_refined = None
        relaxations = [None] * len(self.body_layers)

        for layer_idx, layer in enumerate(self.body_layers):

            if isinstance(layer, Normalization):
                mean_full = layer.mean.detach().to(device=device, dtype=dtype)
                sigma_full = layer.sigma.detach().to(device=device, dtype=dtype)
                if torch.any(sigma_full <= 0):
                    raise ValueError('Normalization sigma must be positive.')
                box_low = (box_low - mean_full) / sigma_full
                box_high = (box_high - mean_full) / sigma_full
                mean_flat = mean_full.expand_as(self.x).reshape(-1)
                sigma_flat = sigma_full.expand_as(self.x).reshape(-1)
                x_l = (x_l - mean_flat) / sigma_flat
                x_u = (x_u - mean_flat) / sigma_flat
                prev_was_relu = False
                continue

            if isinstance(layer, torch.nn.Flatten):
                box_low = box_low.reshape(box_low.size(0), -1)
                box_high = box_high.reshape(box_high.size(0), -1)
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype
                )

                # ---- k-ReLU tightening of this layer's pre-activations ----
                # Only valid when this Linear reads a ReLU's post-activations
                # directly, which is exactly when prev_was_relu holds.
                if (mn is not None and mn.intermediate and prev_was_relu
                        and last_tail is not None):
                    pending_refined = self._intermediate_refinement(
                        mn, last_tail, W, b
                    )

                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
                A_low, A_high = (
                    W_pos @ A_low + W_neg @ A_high,
                    W_pos @ A_high + W_neg @ A_low,
                )
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low, new_b_high = new_b_low + b, new_b_high + b
                b_low, b_high = new_b_low, new_b_high

                box_low_flat = box_low.reshape(box_low.size(0), -1)
                box_high_flat = box_high.reshape(box_high.size(0), -1)
                new_box_low = box_low_flat @ W_pos.t() + box_high_flat @ W_neg.t()
                new_box_high = box_high_flat @ W_pos.t() + box_low_flat @ W_neg.t()
                if b is not None:
                    new_box_low, new_box_high = new_box_low + b, new_box_high + b
                box_low, box_high = new_box_low, new_box_high
                cur_shape = None
                prev_was_relu = False
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    if box_low.dim() != 4:
                        raise RuntimeError(
                            'Conv2d encountered without a valid feature-map shape.'
                        )
                    cur_shape = tuple(box_low.shape[-3:])

                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype
                )
                C_in, H_in, W_in = cur_shape
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                def conv_coeff(weight, coeff):
                    coeff_img = coeff.t().reshape(self.input_dim, C_in, H_in, W_in)
                    return F.conv2d(
                        coeff_img, weight, bias=None, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation,
                        groups=layer.groups,
                    )

                low_img = conv_coeff(W_pos, A_low) + conv_coeff(W_neg, A_high)
                high_img = conv_coeff(W_pos, A_high) + conv_coeff(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                A_low = low_img.reshape(self.input_dim, -1).t()
                A_high = high_img.reshape(self.input_dim, -1).t()

                def conv_bias(weight, bias_vector):
                    bias_img = bias_vector.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(
                        bias_img, weight, bias=None, stride=layer.stride,
                        padding=layer.padding, dilation=layer.dilation,
                        groups=layer.groups,
                    ).reshape(-1)

                new_b_low = conv_bias(W_pos, b_low) + conv_bias(W_neg, b_high)
                new_b_high = conv_bias(W_pos, b_high) + conv_bias(W_neg, b_low)
                if b is not None:
                    expanded = b.view(-1, 1, 1).expand(
                        C_out, H_out, W_out
                    ).reshape(-1)
                    new_b_low, new_b_high = new_b_low + expanded, new_b_high + expanded
                b_low, b_high = new_b_low, new_b_high

                conv_kwargs = dict(
                    bias=None, stride=layer.stride, padding=layer.padding,
                    dilation=layer.dilation, groups=layer.groups,
                )
                new_box_low = (
                    F.conv2d(box_low, W_pos, **conv_kwargs)
                    + F.conv2d(box_high, W_neg, **conv_kwargs)
                )
                new_box_high = (
                    F.conv2d(box_high, W_pos, **conv_kwargs)
                    + F.conv2d(box_low, W_neg, **conv_kwargs)
                )
                if b is not None:
                    bias_img = b.view(1, -1, 1, 1)
                    new_box_low = new_box_low + bias_img
                    new_box_high = new_box_high + bias_img
                box_low, box_high = new_box_low, new_box_high
                cur_shape = (C_out, H_out, W_out)
                prev_was_relu = False
                continue

            if isinstance(layer, torch.nn.ReLU):
                A_low_pos, A_low_neg = A_low.clamp(min=0), A_low.clamp(max=0)
                A_high_pos, A_high_neg = A_high.clamp(min=0), A_high.clamp(max=0)
                affine_l = A_low_pos @ x_l + A_low_neg @ x_u + b_low
                affine_u = A_high_pos @ x_u + A_high_neg @ x_l + b_high

                l = torch.maximum(affine_l, box_low.reshape(-1))
                u = torch.minimum(affine_u, box_high.reshape(-1))

                # ---- DeepPoly backsubstitution -----------------------------
                # This is the main source of precision: it defers every ReLU's
                # line choice until the accumulated coefficient is known.
                back_low = back_high = None
                n_here = l.numel()
                if layer_idx > 0 and self.backsub_affordable(n_here, layer_idx - 1):
                    eye = torch.eye(n_here, device=device, dtype=dtype)
                    zero = torch.zeros(n_here, device=device, dtype=dtype)
                    res_lo = self.backsub(eye, zero, layer_idx - 1, True, relaxations)
                    res_hi = self.backsub(eye, zero, layer_idx - 1, False, relaxations)
                    if res_lo is not None and res_hi is not None:
                        bA_low, bb_low = res_lo
                        bA_high, bb_high = res_hi
                        back_low = (bA_low.clamp(min=0) @ orig_l
                                    + bA_low.clamp(max=0) @ orig_u + bb_low)
                        back_high = (bA_high.clamp(min=0) @ orig_u
                                     + bA_high.clamp(max=0) @ orig_l + bb_high)
                        l = torch.maximum(l, back_low)
                        u = torch.minimum(u, back_high)

                # k-ReLU bounds computed at the preceding affine layer.
                if pending_refined is not None:
                    r_l, r_u = pending_refined
                    if r_l is not None:
                        l = torch.maximum(l, r_l)
                    if r_u is not None:
                        u = torch.minimum(u, r_u)
                    pending_refined = None

                u = torch.maximum(u, l)

                # ---- forced signs from branch and bound -------------------
                entries = splits_by_layer.get(relu_id)
                if entries:
                    neg_ids = [n for n, s in entries if s < 0]
                    pos_ids = [n for n, s in entries if s > 0]

                    if neg_ids:
                        idx = torch.tensor(neg_ids, device=device, dtype=torch.long)
                        if float(l.index_select(0, idx).detach().max()) > FEAS_TOL:
                            return None
                        u = u.index_put((idx,), u.index_select(0, idx).clamp(max=0.0))
                        l = l.index_put(
                            (idx,),
                            torch.minimum(l.index_select(0, idx),
                                          u.index_select(0, idx)),
                        )

                    if pos_ids:
                        idx = torch.tensor(pos_ids, device=device, dtype=torch.long)
                        if float(u.index_select(0, idx).detach().min()) < -FEAS_TOL:
                            return None
                        l = l.index_put((idx,), l.index_select(0, idx).clamp(min=0.0))
                        u = u.index_put(
                            (idx,),
                            torch.maximum(u.index_select(0, idx),
                                          l.index_select(0, idx)),
                        )

                    if float((l - u).detach().max()) > FEAS_TOL:
                        return None

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)

                if initialize_alpha:
                    init = torch.where(u >= -l, torch.ones_like(l), torch.zeros_like(l))
                    alpha_params.append(torch.nn.Parameter(init.detach().clone()))

                if relu_id >= len(alpha_params):
                    raise RuntimeError(
                        'Missing alpha parameters for ReLU layer {}'.format(relu_id)
                    )
                alpha = alpha_params[relu_id]
                if alpha.numel() != l.numel():
                    raise RuntimeError(
                        'Alpha size mismatch at ReLU layer {}'.format(relu_id)
                    )
                this_relu = relu_id
                relu_id += 1

                alpha_safe = alpha.clamp(0.0, 1.0)
                zeros = torch.zeros_like(l)
                ones = torch.ones_like(l)

                lower_scale = torch.where(
                    pos_mask, ones, torch.where(cross_mask, alpha_safe, zeros)
                )
                denom = torch.where(
                    cross_mask, (u - l).clamp_min(self.float_eps), torch.ones_like(u)
                )
                upper_slope = torch.where(cross_mask, u / denom, torch.zeros_like(u))
                upper_scale = torch.where(
                    pos_mask, ones, torch.where(cross_mask, upper_slope, zeros)
                )
                upper_intercept = torch.where(cross_mask, -upper_slope * l, zeros)

                relaxations[layer_idx] = (lower_scale, upper_scale, upper_intercept)

                # Pre-activation state, kept for the joint relaxation.  Prefer
                # the backsubstituted symbolic expressions when we have them:
                # they are tighter, so the octahedral group bounds derived
                # from them are tighter too.
                if back_low is not None:
                    tail_A_low, tail_b_low = bA_low, bb_low
                    tail_A_high, tail_b_high = bA_high, bb_high
                    tail_l, tail_u = orig_l, orig_u
                else:
                    tail_A_low, tail_b_low = A_low, b_low
                    tail_A_high, tail_b_high = A_high, b_high
                    tail_l, tail_u = x_l, x_u

                last_tail = {
                    'index': this_relu,
                    'A_low': tail_A_low, 'b_low': tail_b_low,
                    'A_high': tail_A_high, 'b_high': tail_b_high,
                    'l': l, 'u': u, 'cross': cross_mask,
                    'lower_scale': lower_scale, 'upper_scale': upper_scale,
                    'upper_intercept': upper_intercept,
                    'x_l': tail_l, 'x_u': tail_u,
                }

                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low
                A_high = upper_scale.unsqueeze(1) * A_high
                b_high = upper_scale * b_high + upper_intercept

                if collect_info:
                    low_slack = torch.zeros_like(l, requires_grad=True)
                    high_slack = torch.zeros_like(l, requires_grad=True)
                    b_low = b_low + low_slack
                    b_high = b_high + high_slack
                    info['relu'].append({
                        'l': l.detach(), 'u': u.detach(),
                        'alpha': alpha_safe.detach(), 'cross': cross_mask.detach(),
                        'intercept': upper_intercept.detach(),
                        'low_slack': low_slack, 'high_slack': high_slack,
                    })

                box_low = torch.clamp(l, min=0).reshape_as(box_low)
                box_high = torch.clamp(u, min=0).reshape_as(box_high)
                prev_was_relu = True
                continue

            raise NotImplementedError(
                'Unsupported layer type before output: {}'.format(type(layer))
            )

        W_out = self.output_layer.weight.detach().to(device=device, dtype=dtype)
        if self.output_layer.bias is None:
            b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype)
        else:
            b_out = self.output_layer.bias.detach().to(device=device, dtype=dtype)

        if target_positions is None:
            selected = self.other_labels
        else:
            selected = self.other_labels.index_select(0, target_positions)

        W_margin = W_out[self.true_label].unsqueeze(0) - W_out.index_select(0, selected)
        b_margin = b_out[self.true_label] - b_out.index_select(0, selected)
        W_pos, W_neg = W_margin.clamp(min=0), W_margin.clamp(max=0)

        margin_A = W_pos @ A_low + W_neg @ A_high
        margin_b = W_pos @ b_low + W_neg @ b_high + b_margin
        affine_margin = (
            margin_A.clamp(min=0) @ x_l + margin_A.clamp(max=0) @ x_u + margin_b
        )
        box_margin = (
            W_pos @ box_low.reshape(-1) + W_neg @ box_high.reshape(-1) + b_margin
        )
        margins = torch.maximum(affine_margin, box_margin)

        # ---- DeepPoly backsubstitution of the margin ----------------------
        # Only a handful of rows, so this is always affordable and it is where
        # the deferred line choice pays off most.
        if self.body_layers:
            res = self.backsub(
                W_margin, b_margin, len(self.body_layers) - 1, True, relaxations
            )
            if res is not None:
                bA, bk = res
                back_margin = (bA.clamp(min=0) @ orig_l
                               + bA.clamp(max=0) @ orig_u + bk)
                margins = torch.maximum(margins, back_margin)

        # ---- k-ReLU joint relaxation at the output margin ------------------
        if (mn is not None and mn.margin_groups is not None
                and self.margin_refinable and last_tail is not None
                and W_margin.size(0) == mn.margin_d.size(0)):
            refined = _refine_linear_forms(
                W_margin, b_margin, mn.margin_groups, mn.margin_d,
                last_tail, last_tail['x_l'], last_tail['x_u'],
            )
            if refined is not None:
                margins = torch.maximum(margins, refined)

        extra = info if collect_info else alpha_params
        if capture_tail:
            return margins, extra, last_tail, x_l, x_u
        return margins, extra

    def _intermediate_refinement(self, mn, tail, W, b):
        """Tighter pre-activation bounds for  W @ ReLU(z) + b  via k-ReLU.

        Uses the DeepPoly-equivalent d, so no extra parameters are introduced
        and the result is guaranteed to be at least as tight as DeepPoly.
        """
        x_l, x_u = tail['x_l'], tail['x_u']
        key = tail['index']
        groups = mn.layer_groups.get(key)
        if groups is None:
            with torch.no_grad():
                groups = self.build_layer_groups(tail, x_l, x_u)
            mn.layer_groups[key] = groups if groups is not None else False
        if groups is False or groups is None:
            return None
        if W.size(1) != tail['l'].numel():
            return None

        bias = b if b is not None else torch.zeros(
            W.size(0), device=W.device, dtype=W.dtype
        )
        c_rows = torch.cat([W, -W], dim=0)
        const = torch.cat([bias, -bias], dim=0)

        d = _deeppoly_d(
            c_rows, groups, tail['lower_scale'], tail['upper_scale']
        )
        out = _refine_linear_forms(c_rows, const, groups, d, tail, x_l, x_u)
        if out is None:
            return None
        m = W.size(0)
        return out[:m], -out[m:]

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _clone_params(values, inset=None):
        params = []
        for value in values:
            init = value.detach().clone()
            if inset is not None:
                init.clamp_(inset, 1.0 - inset)
            params.append(torch.nn.Parameter(init))
        return params

    @staticmethod
    def _project(params):
        with torch.no_grad():
            for p in params:
                p.clamp_(0.0, 1.0)

    def make_mn_context(self, alpha_values, splits, x_l, x_u, target_pos,
                        intermediate):
        """Pick groups and initialise d for one target class."""
        mn = _MultiNeuronContext(intermediate=intermediate)
        if not self.margin_refinable:
            return mn

        params = self._clone_params(alpha_values)
        target_index = torch.tensor([target_pos], device=self.device, dtype=torch.long)
        with torch.no_grad():
            result = self.propagate(
                alpha_params=params, splits=splits, x_l=x_l, x_u=x_u,
                target_positions=target_index, capture_tail=True,
            )
        if result is None:
            return mn
        _, _, tail, _, _ = result
        if tail is None:
            return mn
        tail_l, tail_u = tail['x_l'], tail['x_u']

        W_out = self.output_layer.weight.detach()
        selected = self.other_labels[target_pos]
        c_row = (W_out[self.true_label] - W_out[selected]).to(
            device=self.device, dtype=self.dtype
        )
        if c_row.numel() != tail['l'].numel():
            return mn

        with torch.no_grad():
            groups = self.build_margin_groups(tail, c_row, tail_l, tail_u)
        if groups is None:
            return mn

        with torch.no_grad():
            d0 = _deeppoly_d(
                c_row.unsqueeze(0), groups,
                tail['lower_scale'], tail['upper_scale'],
            )
        mn.margin_groups = groups
        mn.margin_d = torch.nn.Parameter(d0.detach().clone())
        return mn

    def optimise_target(self, alpha_values, splits, x_l, x_u, target_pos,
                        steps, lr, deadline, multi_neuron=True,
                        intermediate=False):
        """Maximise the bound for one target class. Returns (bound, alphas)."""
        params = self._clone_params(alpha_values)
        target_index = torch.tensor([target_pos], device=self.device, dtype=torch.long)

        mn = None
        if multi_neuron:
            try:
                mn = self.make_mn_context(
                    alpha_values, splits, x_l, x_u, target_pos, intermediate
                )
                if mn.margin_groups is None and not mn.intermediate:
                    mn = None
            except Exception:
                mn = None

        def evaluate():
            try:
                return self.propagate(
                    alpha_params=params, splits=splits, x_l=x_l, x_u=x_u,
                    target_positions=target_index, mn=mn,
                )
            except Exception:
                # Any numerical trouble in the joint relaxation degrades to
                # plain DeepPoly rather than losing the case.
                return self.propagate(
                    alpha_params=params, splits=splits, x_l=x_l, x_u=x_u,
                    target_positions=target_index,
                )

        result = evaluate()
        if result is None:
            return None, None
        best = float(result[0][0].detach())
        best_values = [p.detach().clone() for p in params]
        if best > CERT_TOL:
            return best, best_values

        groups = [{'params': params, 'lr': lr}]
        if mn is not None and mn.margin_d is not None:
            groups.append({'params': [mn.margin_d], 'lr': self.MN_D_LR})
        optimizer = torch.optim.Adam(groups)

        for _ in range(steps):
            if time.perf_counter() >= deadline:
                break
            optimizer.zero_grad(set_to_none=True)
            result = evaluate()
            if result is None:
                return None, None
            margin = result[0][0]
            value = float(margin.detach())
            if value > best:
                best = value
                best_values = [p.detach().clone() for p in params]
            if best > CERT_TOL or not torch.isfinite(margin):
                break
            (-margin).backward()
            optimizer.step()
            self._project(params)

        return best, best_values

    def branch_scores(self, alpha_values, splits, x_l, x_u, target_pos):
        """Rank unstable neurons by their share of the lost margin."""
        params = self._clone_params(alpha_values)
        target_index = torch.tensor([target_pos], device=self.device, dtype=torch.long)
        result = self.propagate(
            alpha_params=params, splits=splits, x_l=x_l, x_u=x_u,
            target_positions=target_index, collect_info=True,
        )
        if result is None:
            return None
        margins, info = result
        margin = margins[0]
        if not torch.isfinite(margin):
            return None
        margin.backward()

        scores, fallback = [], []
        for record in info['relu']:
            l, u = record['l'], record['u']
            alpha = record['alpha']
            cross = record['cross']
            lower_gap = torch.maximum((1.0 - alpha) * u, -alpha * l).clamp_min(0.0)
            area = record['intercept'].clamp_min(0.0) + lower_gap
            fallback.append(torch.where(cross, area, torch.zeros_like(area)))

            g_low = record['low_slack'].grad
            g_high = record['high_slack'].grad
            if g_low is None or g_high is None:
                scores.append(torch.zeros_like(area))
                continue
            score = (g_high.abs() * record['intercept'].clamp_min(0.0)
                     + g_low.abs() * lower_gap)
            scores.append(torch.where(cross, score, torch.zeros_like(score)))

        if all((s.numel() == 0 or float(s.max()) <= 0.0) for s in scores):
            return fallback
        return scores

    def pick_branch_neuron(self, scores, splits):
        best_key, best_score = None, 0.0
        for layer_idx, score in enumerate(scores):
            if score.numel() == 0:
                continue
            masked = score.clone()
            for (l_idx, neuron), _ in splits.items():
                if l_idx == layer_idx:
                    masked[neuron] = -1.0
            value, index = masked.max(dim=0)
            value = float(value)
            if value > best_score:
                best_score = value
                best_key = (layer_idx, int(index))
        return best_key

    # -- stage 3 ------------------------------------------------------------

    def branch_and_bound(self, target_pos, alpha_values, root_bound, deadline):
        counter = itertools.count()
        heap = [(root_bound, next(counter), {}, alpha_values)]

        while heap:
            if time.perf_counter() >= deadline:
                return False
            if len(heap) > self.MAX_OPEN_SUBPROBLEMS:
                return False

            bound, _, splits, alphas = heapq.heappop(heap)
            if bound > CERT_TOL:
                return True
            if len(splits) >= self.MAX_SPLIT_DEPTH:
                return False

            parent_l, parent_u, feasible = self.tighten_input_box(splits)
            if not feasible:
                continue

            scores = self.branch_scores(alphas, splits, parent_l, parent_u, target_pos)
            if scores is None:
                continue
            key = self.pick_branch_neuron(scores, splits)
            if key is None:
                return False

            for sign in (-1, 1):
                if time.perf_counter() >= deadline:
                    return False

                child_splits = dict(splits)
                child_splits[key] = sign

                child_l, child_u, feasible = self.tighten_input_box(child_splits)
                if not feasible:
                    continue

                child_bound, child_alphas = self.optimise_target(
                    alphas, child_splits, child_l, child_u, target_pos,
                    self.SUB_STEPS, self.SUB_LR, deadline,
                    multi_neuron=True, intermediate=False,
                )
                if child_bound is None:
                    continue

                # The parent's bound holds over a superset of this branch, so
                # it stays valid.  Keeping it matters: DeepPoly's final
                # concretisation exploits cancellation between neuron rows, so
                # zeroing one row can lower the raw child bound even though
                # the child's feasible set is smaller.
                child_bound = max(child_bound, bound)

                if child_bound > CERT_TOL:
                    continue
                heapq.heappush(
                    heap, (child_bound, next(counter), child_splits, child_alphas)
                )

        return True

    # -- driver -------------------------------------------------------------

    def run(self, deadline):
        # ---- Stage 1a: DeepPoly with the 0/1 slope heuristic --------------
        params = []
        with torch.enable_grad():
            margins, params = self.propagate(
                alpha_params=params, initialize_alpha=True
            )
        certified = margins.detach() > CERT_TOL
        if bool(certified.all()):
            return True
        if not params:
            return False

        # ---- Stage 1b: one shared slope set for all targets --------------
        shared = self._clone_params(params, inset=self.INIT_INSET)
        optimizer = torch.optim.Adam(shared, lr=self.ROOT_LR)
        with torch.enable_grad():
            for _ in range(self.SHARED_STEPS):
                if time.perf_counter() >= deadline:
                    break
                optimizer.zero_grad(set_to_none=True)
                margins, _ = self.propagate(alpha_params=shared)
                certified = certified | (margins.detach() > CERT_TOL)
                if bool(certified.all()):
                    return True
                unresolved = margins[~certified]
                if unresolved.numel() == 0:
                    return True
                loss = self.TEMPERATURE * torch.logsumexp(
                    -unresolved / self.TEMPERATURE, dim=0
                )
                if not torch.isfinite(loss):
                    break
                loss.backward()
                optimizer.step()
                self._project(shared)

        with torch.no_grad():
            margins, _ = self.propagate(alpha_params=shared)
            certified = certified | (margins > CERT_TOL)
            if bool(certified.all()):
                return True

        shared_values = [p.detach().clone() for p in shared]
        remaining = (~certified).nonzero(as_tuple=True)[0].tolist()

        # ---- Stages 1c + 2: per-target slopes and k-ReLU -----------------
        still_open = []
        per_target_alphas = {}
        per_target_bound = {}
        for target_pos in remaining:
            if time.perf_counter() >= deadline:
                return False
            bound, values = self.optimise_target(
                shared_values, None, self.base_l, self.base_u, target_pos,
                self.TARGET_STEPS, self.ROOT_LR, deadline,
                multi_neuron=True, intermediate=not self.has_conv,
            )
            if bound is None:
                continue
            if bound > CERT_TOL:
                continue
            still_open.append(target_pos)
            per_target_alphas[target_pos] = values
            per_target_bound[target_pos] = bound

        if not still_open:
            return True

        # ---- Stage 3: branch and bound on what is left -------------------
        # Hardest target first: every target has to be proved, so if the
        # hardest one is hopeless we find out before spending the budget on
        # the easy ones.  It also gets nearly all the remaining time -- an
        # even split would starve exactly the target that needs the search.
        still_open.sort(key=lambda pos: per_target_bound[pos])
        for index, target_pos in enumerate(still_open):
            now = time.perf_counter()
            if now >= deadline:
                return False
            reserve = self.TARGET_RESERVE * (len(still_open) - index - 1)
            budget = max(deadline - now - reserve, 1.0)
            ok = self.branch_and_bound(
                target_pos, per_target_alphas[target_pos],
                per_target_bound[target_pos], now + budget,
            )
            if not ok:
                return False
        return True


def analyze(model, x, eps, true_label):
    deadline = time.perf_counter() + TIME_BUDGET

    attack_deadline = time.perf_counter() + PGD_BUDGET
    if _pgd_falsify(model, x, eps, true_label, deadline=attack_deadline):
        return False

    verifier = _Verifier(model, x, eps, true_label)
    return verifier.run(deadline)


def load_network(net_name):
    if net_name == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif net_name == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif net_name == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif net_name == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif net_name == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif net_name == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif net_name == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        raise ValueError(f'Unsupported net name: {net_name}')

    net.load_state_dict(torch.load(f'../mnist_nets/{net_name}.pt', map_location=torch.device(DEVICE)))
    return net


def parse_spec(spec_path):
    with open(spec_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]

    true_label = int(lines[0])
    pixel_values = [float(line) for line in lines[1:]]
    eps = float(spec_path[:-4].split('/')[-1].split('_')[-1])
    return true_label, pixel_values, eps


def run_single_case(net_name, spec_path):
    true_label, pixel_values, eps = parse_spec(spec_path)
    net = load_network(net_name)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    return analyze(net, inputs, eps, true_label)


def run_all_cases(test_dir):
    base_dir = os.path.abspath(test_dir)

    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(base_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue

        spec_paths = sorted(
            os.path.join(net_dir, filename)
            for filename in os.listdir(net_dir)
            if filename.endswith('.txt')
        )

        for spec_path in spec_paths:
            start = time.perf_counter()
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue

            status = 'verified' if result else 'not verified'
            elapsed = time.perf_counter() - start
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}\t{elapsed:.1f}s')


# run single case: python verifier.py --net fc1 --spec ../test_cases/fc1/img0_0.09500.txt
# run all cases: python verifier.py --batch --tests-dir ../test_cases

def main():
    parser = argparse.ArgumentParser(description='Neural network verification')
    parser.add_argument('--net', type=str, choices=NETWORK_NAMES, help='Neural network architecture')
    parser.add_argument('--spec', type=str, help='Test case to verify')
    parser.add_argument('--batch', action='store_true', help='Run all test cases under the tests directory')
    parser.add_argument('--tests-dir', type=str, default='../test_cases', help='Directory with the test case folders')
    args = parser.parse_args()

    if args.batch:
        run_all_cases(args.tests_dir)
        return

    if args.net is None or args.spec is None:
        parser.error('Please provide both --net and --spec, or use --batch.')

    true_label, pixel_values, eps = parse_spec(args.spec)
    net = load_network(args.net)

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')


if __name__ == '__main__':
    main()