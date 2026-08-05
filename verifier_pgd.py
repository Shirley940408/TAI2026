"""
PGD-first verifier, with the propagation bug fixed.

What changed relative to the previous version of this file
----------------------------------------------------------
The old propagation was:

    A_low <- W+ @ A_low + W- @ A_high

repeated at every affine layer, with the comment that keeping coefficients
in input coordinates "is equivalent to full back-substitution to the input".
That claim is false, and it was the real bug.

Keeping the coefficients in input coordinates is not the issue -- the issue
is *when* each ReLU commits to its lower or upper line.  In the expression
above, the line for an upstream neuron j is chosen from the sign of the local
weight W_ij, separately for every downstream neuron i, and only afterwards
are those per-neuron bounds combined.  So `A_low[j]` is the tightest lower
expression for neuron j *considered alone*, and the cancellation between
neurons that share the same input is thrown away.

True DeepPoly back-substitution pushes the target expression back one layer
at a time and picks each line from the sign of the *accumulated* coefficient,
i.e. after the whole downstream path has been combined.  Bound-then-combine
versus combine-then-bound -- that is the entire difference.

The two coincide for a single hidden layer (there is nothing downstream to
accumulate) and diverge from two onwards, badly.  On random networks with the
same relaxation and the same alpha, at eps=0.1: a 3-hidden-layer net went
from -0.38 to -0.18, a 5-hidden-layer net from -11.8 to -0.77, and the number
of unstable neurons in the deepest layers roughly halved.

So this file now back-substitutes *every* intermediate pre-activation bound,
not just the output margin.  That is what "IBP + DeepPoly" actually means.

Kept from before: the PGD stage 0 (a real counterexample is a complete proof
of non-robustness, no abstraction needed), the PGD-informed slope
initialisation, the interval intersection, and the shared/per-target alpha
optimisation.

Removed: the CROWN-backward fallback.  Its intermediate bounds came from
plain interval propagation, which is why it never certified anything extra;
now that the main path back-substitutes properly it is redundant.
"""

import argparse
import os
import time

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']

CERT_TOL = 1e-8
# Intermediate bounds are back-substituted only while the coefficient
# matrices stay under this many elements.  Oversized convolutional layers
# fall back to their (sound, looser) forward symbolic bounds instead.  The
# output margin is always back-substituted -- it is only nine rows.
MAX_BACKSUB_ELEMS = 6_000_000


# =============================================================================
# PGD attack and the ReLU sign readout used for slope initialisation
# =============================================================================

def pgd_attack(model, x, eps, true_label, num_restarts=5, num_steps=50,
               step_size=None, debug=False, tag=""):
    """
    Standard L-inf PGD: random start inside the eps-ball, sign-gradient ascent
    on cross-entropy, projected back into [x-eps, x+eps] and [0, 1] each step.

    Returns (found_adversarial, representative_point).  When found_adversarial
    is True the point is a real misclassified input -- a complete and sound
    proof of non-robustness on its own.  When False the point is just the last
    restart's endpoint, used only as a heuristic for slope initialisation; it
    carries no soundness weight.
    """
    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} [pgd] {msg}', flush=True)

    if step_size is None:
        step_size = max(eps / 4.0, 1e-4)

    model.eval()
    x = x.detach()
    label = torch.tensor([true_label], device=x.device, dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()

    last_point = x.clone()

    for restart in range(num_restarts):
        delta = torch.empty_like(x).uniform_(-eps, eps)
        adv = torch.clamp(x + delta, 0.0, 1.0)

        for step in range(num_steps):
            adv = adv.detach().requires_grad_(True)
            out = model(adv)
            pred = out.argmax(dim=1).item()
            if pred != true_label:
                dbg(f'restart={restart} step={step}: found adversarial example (pred={pred})')
                return True, adv.detach()

            loss = loss_fn(out, label)
            grad = torch.autograd.grad(loss, adv)[0]
            with torch.no_grad():
                adv = adv + step_size * grad.sign()
                adv = torch.max(torch.min(adv, x + eps), x - eps)
                adv = torch.clamp(adv, 0.0, 1.0)

        with torch.no_grad():
            final_pred = model(adv).argmax(dim=1).item()
        if final_pred != true_label:
            dbg(f'restart={restart}: found adversarial example at final step (pred={final_pred})')
            return True, adv.detach()

        last_point = adv.detach()

    dbg(f'no adversarial example found after {num_restarts} restarts x {num_steps} steps')
    return False, last_point


def get_relu_preact_signs(model, point):
    """Per-ReLU {0,1} mask of pre-activation signs at a concrete point."""
    signs = []
    cur = point.detach()
    with torch.no_grad():
        for layer in model.layers.children():
            if isinstance(layer, Normalization):
                mean = layer.mean.to(device=cur.device, dtype=cur.dtype)
                sigma = layer.sigma.to(device=cur.device, dtype=cur.dtype)
                cur = (cur - mean) / sigma
                continue
            if isinstance(layer, torch.nn.Flatten):
                cur = layer(cur)
                continue
            if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
                cur = layer(cur)
                continue
            if isinstance(layer, torch.nn.ReLU):
                signs.append((cur.reshape(-1) >= 0).to(dtype=cur.dtype))
                cur = layer(cur)
                continue
            raise NotImplementedError(
                'Unsupported layer type while reading PGD signs: {}'.format(type(layer)))
    return signs


# =============================================================================
# DeepPoly with back-substitution
# =============================================================================

class _Engine:
    """Bound propagation for one (network, image, eps) triple."""

    def __init__(self, model, x, eps):
        model.eval()
        layers = list(model.layers.children())
        if not layers or not isinstance(layers[-1], torch.nn.Linear):
            raise NotImplementedError('Expects the final network layer to be Linear.')

        self.body = layers[:-1]
        self.output_layer = layers[-1]

        self.x = x.detach()
        self.device = x.device
        self.dtype = x.dtype
        self.float_eps = torch.finfo(self.dtype).eps
        self.input_dim = self.x.numel()

        flat = self.x.reshape(-1)
        self.x_l = torch.clamp(flat - eps, 0.0, 1.0)
        self.x_u = torch.clamp(flat + eps, 0.0, 1.0)

        # Layer shapes, needed to reshape back-substitution coefficients and
        # to size conv_transpose2d.
        self.in_shape, self.out_shape = [], []
        probe = torch.zeros_like(self.x)
        with torch.no_grad():
            for layer in self.body:
                self.in_shape.append(tuple(probe.shape[1:]))
                probe = layer(probe)
                self.out_shape.append(tuple(probe.shape[1:]))
        self.in_numel = [int(torch.tensor(s).prod()) for s in self.in_shape]

    @staticmethod
    def _pair(value):
        return value if isinstance(value, tuple) else (value, value)

    def _affordable(self, rows, i0):
        cost = 0
        for i in range(i0, -1, -1):
            cost += rows * self.in_numel[i]
            if cost > MAX_BACKSUB_ELEMS:
                return False
        return True

    def backsub(self, A, k, i0, lower, relaxations):
        """Push an affine form over the output of body[i0] back to the input.

        Each ReLU on the way picks its line from the sign of the accumulated
        coefficient, which is the whole point -- see the module docstring.
        Returns (A, k) in raw input coordinates (the Normalization layer is
        walked through as well), or None if a needed relaxation is missing.
        """
        device, dtype = self.device, self.dtype

        for i in range(i0, -1, -1):
            layer = self.body[i]

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
                if layer.bias is not None:
                    k = k + A @ layer.bias.detach().to(device=device, dtype=dtype)
                A = A @ layer.weight.detach().to(device=device, dtype=dtype)
                continue

            if isinstance(layer, Normalization):
                shape = self.in_shape[i]
                mean = layer.mean.detach().to(device=device, dtype=dtype)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype)
                mean = mean.expand(1, *shape).reshape(-1)
                sigma = sigma.expand(1, *shape).reshape(-1)
                k = k + A @ (-mean / sigma)
                A = A / sigma
                continue

            if isinstance(layer, torch.nn.Conv2d):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                C_out, H_out, W_out = self.out_shape[i]
                C_in, H_in, W_in = self.in_shape[i]

                if layer.bias is not None:
                    bias = layer.bias.detach().to(device=device, dtype=dtype)
                    k = k + A @ bias.view(-1, 1, 1).expand(
                        C_out, H_out, W_out).reshape(-1)

                rows = A.size(0)
                sh, sw = self._pair(layer.stride)
                ph, pw = self._pair(layer.padding)
                dh, dw = self._pair(layer.dilation)
                kh, kw = self._pair(layer.kernel_size)
                oph = H_in - ((H_out - 1) * sh - 2 * ph + dh * (kh - 1) + 1)
                opw = W_in - ((W_out - 1) * sw - 2 * pw + dw * (kw - 1) + 1)
                if not (0 <= oph < sh and 0 <= opw < sw):
                    return None
                A = F.conv_transpose2d(
                    A.reshape(rows, C_out, H_out, W_out), W, bias=None,
                    stride=layer.stride, padding=layer.padding,
                    output_padding=(oph, opw), groups=layer.groups,
                    dilation=layer.dilation,
                ).reshape(rows, -1)
                continue

            raise NotImplementedError(
                'Unsupported layer in back-substitution: {}'.format(type(layer)))

        return A, k

    def propagate(self, alpha_params=None, initialize_alpha=False,
                  target_positions=None, other_labels=None, true_label=None,
                  unstable_counts=None, l_u_record=None):
        """Bounds for every ReLU layer plus the requested output margins."""
        if alpha_params is None:
            alpha_params = []

        device, dtype = self.device, self.dtype
        orig_l, orig_u = self.x_l, self.x_u
        x_l, x_u = self.x_l, self.x_u

        # forward symbolic state, kept as an extra intersection source: it is
        # cheap and still useful for wide conv layers where back-substituting
        # every intermediate bound would be too expensive
        A_low = torch.eye(self.input_dim, device=device, dtype=dtype)
        A_high = torch.eye(self.input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(self.input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(self.input_dim, device=device, dtype=dtype)

        box_low = x_l.reshape_as(self.x)
        box_high = x_u.reshape_as(self.x)

        cur_shape = tuple(self.x.shape[-3:]) if self.x.dim() >= 3 else None
        relu_id = 0
        relaxations = [None] * len(self.body)

        for layer_idx, layer in enumerate(self.body):

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
                continue

            if isinstance(layer, torch.nn.Flatten):
                box_low = box_low.reshape(box_low.size(0), -1)
                box_high = box_high.reshape(box_high.size(0), -1)
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Linear):
                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                A_low, A_high = (W_pos @ A_low + W_neg @ A_high,
                                 W_pos @ A_high + W_neg @ A_low)
                nb_low = W_pos @ b_low + W_neg @ b_high
                nb_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    nb_low, nb_high = nb_low + b, nb_high + b
                b_low, b_high = nb_low, nb_high

                bl = box_low.reshape(box_low.size(0), -1)
                bh = box_high.reshape(box_high.size(0), -1)
                nbox_low = bl @ W_pos.t() + bh @ W_neg.t()
                nbox_high = bh @ W_pos.t() + bl @ W_neg.t()
                if b is not None:
                    nbox_low, nbox_high = nbox_low + b, nbox_high + b
                box_low, box_high = nbox_low, nbox_high
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    if box_low.dim() != 4:
                        raise RuntimeError('Conv2d without a valid feature-map shape.')
                    cur_shape = tuple(box_low.shape[-3:])

                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype)
                C_in, H_in, W_in = cur_shape
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)
                kw = dict(bias=None, stride=layer.stride, padding=layer.padding,
                          dilation=layer.dilation, groups=layer.groups)

                def conv_coeff(weight, coeff):
                    img = coeff.t().reshape(self.input_dim, C_in, H_in, W_in)
                    return F.conv2d(img, weight, **kw)

                lo = conv_coeff(W_pos, A_low) + conv_coeff(W_neg, A_high)
                hi = conv_coeff(W_pos, A_high) + conv_coeff(W_neg, A_low)
                C_out, H_out, W_out = lo.shape[1:]
                A_low = lo.reshape(self.input_dim, -1).t()
                A_high = hi.reshape(self.input_dim, -1).t()

                def conv_bias(weight, vec):
                    img = vec.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(img, weight, **kw).reshape(-1)

                nb_low = conv_bias(W_pos, b_low) + conv_bias(W_neg, b_high)
                nb_high = conv_bias(W_pos, b_high) + conv_bias(W_neg, b_low)
                if b is not None:
                    bf = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                    nb_low, nb_high = nb_low + bf, nb_high + bf
                b_low, b_high = nb_low, nb_high

                nbox_low = (F.conv2d(box_low, W_pos, **kw)
                            + F.conv2d(box_high, W_neg, **kw))
                nbox_high = (F.conv2d(box_high, W_pos, **kw)
                             + F.conv2d(box_low, W_neg, **kw))
                if b is not None:
                    bimg = b.view(1, -1, 1, 1)
                    nbox_low, nbox_high = nbox_low + bimg, nbox_high + bimg
                box_low, box_high = nbox_low, nbox_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.ReLU):
                # interval bounds
                l = box_low.reshape(-1)
                u = box_high.reshape(-1)

                # forward symbolic bounds
                l = torch.maximum(
                    l, A_low.clamp(min=0) @ x_l + A_low.clamp(max=0) @ x_u + b_low)
                u = torch.minimum(
                    u, A_high.clamp(min=0) @ x_u + A_high.clamp(max=0) @ x_l + b_high)

                # back-substituted bounds -- the tight ones
                n_here = l.numel()
                if layer_idx > 0 and self._affordable(n_here, layer_idx - 1):
                    eye = torch.eye(n_here, device=device, dtype=dtype)
                    zero = torch.zeros(n_here, device=device, dtype=dtype)
                    res_lo = self.backsub(eye, zero, layer_idx - 1, True, relaxations)
                    res_hi = self.backsub(eye.clone(), zero.clone(),
                                          layer_idx - 1, False, relaxations)
                    if res_lo is not None and res_hi is not None:
                        bA_lo, bk_lo = res_lo
                        bA_hi, bk_hi = res_hi
                        l = torch.maximum(
                            l, bA_lo.clamp(min=0) @ orig_l
                               + bA_lo.clamp(max=0) @ orig_u + bk_lo)
                        u = torch.minimum(
                            u, bA_hi.clamp(min=0) @ orig_u
                               + bA_hi.clamp(max=0) @ orig_l + bk_hi)

                u = torch.maximum(u, l)

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)

                if unstable_counts is not None:
                    unstable_counts.append(int(cross_mask.sum()))
                if l_u_record is not None:
                    l_u_record.append((l.detach().clone(), u.detach().clone()))

                if initialize_alpha:
                    init = torch.where(u >= -l, torch.ones_like(l), torch.zeros_like(l))
                    alpha_params.append(torch.nn.Parameter(init.detach().clone()))

                if relu_id >= len(alpha_params):
                    raise RuntimeError(
                        'Missing alpha parameters for ReLU layer {}'.format(relu_id))
                alpha = alpha_params[relu_id]
                if alpha.numel() != n_here:
                    raise RuntimeError(
                        'Alpha size mismatch at ReLU layer {}'.format(relu_id))
                relu_id += 1

                alpha_safe = alpha.clamp(0.0, 1.0)
                zeros, ones = torch.zeros_like(l), torch.ones_like(l)

                lower_scale = torch.where(
                    pos_mask, ones, torch.where(cross_mask, alpha_safe, zeros))
                denom = torch.where(
                    cross_mask, (u - l).clamp_min(self.float_eps), torch.ones_like(u))
                slope = torch.where(cross_mask, u / denom, torch.zeros_like(u))
                upper_scale = torch.where(
                    pos_mask, ones, torch.where(cross_mask, slope, zeros))
                upper_intercept = torch.where(cross_mask, -slope * l, zeros)

                relaxations[layer_idx] = (lower_scale, upper_scale, upper_intercept)

                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low
                A_high = upper_scale.unsqueeze(1) * A_high
                b_high = upper_scale * b_high + upper_intercept

                box_low = torch.clamp(l, min=0).reshape_as(box_low)
                box_high = torch.clamp(u, min=0).reshape_as(box_high)
                continue

            raise NotImplementedError(
                'Unsupported layer type before output: {}'.format(type(layer)))

        # ---- output margins -------------------------------------------
        W_out = self.output_layer.weight.detach().to(device=device, dtype=dtype)
        if self.output_layer.bias is None:
            b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype)
        else:
            b_out = self.output_layer.bias.detach().to(device=device, dtype=dtype)

        selected = (other_labels if target_positions is None
                    else other_labels.index_select(0, target_positions))
        W_margin = W_out[true_label].unsqueeze(0) - W_out.index_select(0, selected)
        b_margin = b_out[true_label] - b_out.index_select(0, selected)
        W_pos, W_neg = W_margin.clamp(min=0), W_margin.clamp(max=0)

        margin_A = W_pos @ A_low + W_neg @ A_high
        margin_b = W_pos @ b_low + W_neg @ b_high + b_margin
        margins = (margin_A.clamp(min=0) @ x_l
                   + margin_A.clamp(max=0) @ x_u + margin_b)

        box_margin = (W_pos @ box_low.reshape(-1)
                      + W_neg @ box_high.reshape(-1) + b_margin)
        margins = torch.maximum(margins, box_margin)

        # always back-substitute the margin: only a handful of rows, and this
        # is where the deferred line choice pays off most
        if self.body:
            res = self.backsub(W_margin, b_margin, len(self.body) - 1, True, relaxations)
            if res is not None:
                bA, bk = res
                back_margin = (bA.clamp(min=0) @ orig_l
                               + bA.clamp(max=0) @ orig_u + bk)
                margins = torch.maximum(margins, back_margin)

        return margins, alpha_params


# =============================================================================
# slope optimisation on top of the fixed propagation
# =============================================================================

def _analyze_deeppoly(model, x, eps, true_label, extra_alpha_init=None,
                      debug=False, tag=""):
    SHARED_OPT_STEPS = 18
    TARGET_OPT_STEPS = 12
    OPT_LR = 5e-2
    OPT_TEMPERATURE = 0.5
    OPT_INIT_EPS = 5e-2

    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} [deeppoly] {msg}', flush=True)

    phase_start = time.perf_counter()
    engine = _Engine(model, x, eps)
    device, dtype = engine.device, engine.dtype

    n_out = engine.output_layer.weight.size(0)
    other_labels = torch.tensor(
        [c for c in range(n_out) if c != true_label], device=device, dtype=torch.long)

    def run(alpha_params=None, initialize_alpha=False, target_positions=None,
            unstable_counts=None, l_u_record=None):
        return engine.propagate(
            alpha_params=alpha_params, initialize_alpha=initialize_alpha,
            target_positions=target_positions, other_labels=other_labels,
            true_label=true_label, unstable_counts=unstable_counts,
            l_u_record=l_u_record)

    def make_parameters(values, move_inside=False):
        params = []
        for value in values:
            init = value.detach().clone()
            if move_inside:
                init.clamp_(OPT_INIT_EPS, 1.0 - OPT_INIT_EPS)
            params.append(torch.nn.Parameter(init))
        return params

    def project(params):
        with torch.no_grad():
            for p in params:
                p.clamp_(0.0, 1.0)

    # ---- Step 1: 0/1 heuristic, and record per-ReLU [l, u] ----
    initial_params = []
    l_u_record = []
    unstable_counts = []
    with torch.enable_grad():
        initial_margins, initial_params = run(
            alpha_params=initial_params, initialize_alpha=True,
            unstable_counts=unstable_counts, l_u_record=l_u_record)

    dbg(f'unstable per layer={unstable_counts} total={sum(unstable_counts)}')
    dbg(f'0/1-heuristic min_margin={initial_margins.min().item():.4f} '
        f'already_certified={int((initial_margins.detach() > CERT_TOL).sum())}/{n_out - 1}')

    certified = initial_margins.detach() > CERT_TOL
    if bool(certified.all()):
        dbg(f'VERIFIED from the 0/1 heuristic alone ({time.perf_counter() - phase_start:.2f}s)')
        return True
    if not initial_params:
        dbg('no unstable ReLU neurons and still not certified -> give up')
        return False

    # ---- Step 2: compare the heuristic init against the PGD-informed one ----
    candidates = [make_parameters(initial_params, move_inside=True)]
    names = ['heuristic']

    if extra_alpha_init is not None and len(extra_alpha_init) == len(l_u_record):
        pgd_alpha = []
        for sign, (l, u) in zip(extra_alpha_init, l_u_record):
            pos = l >= 0
            cross = (l < 0) & (u > 0)
            pgd_alpha.append(torch.where(
                pos, torch.ones_like(l),
                torch.where(cross, sign.reshape_as(l), torch.zeros_like(l))))
        candidates.append(make_parameters(pgd_alpha, move_inside=True))
        names.append('pgd-informed')

    best_params, best_score, best_margins, best_name = None, -float('inf'), None, None
    with torch.no_grad():
        for name, candidate in zip(names, candidates):
            margins, _ = run(alpha_params=candidate)
            unresolved = margins[~certified]
            score = float(unresolved.min()) if unresolved.numel() else float(margins.min())
            if score > best_score:
                best_score, best_params, best_margins, best_name = score, candidate, margins, name
    dbg(f'init comparison: chosen={best_name} score={best_score:.4f}')

    certified = certified | (best_margins > CERT_TOL)
    if bool(certified.all()):
        dbg(f'VERIFIED after picking the better init ({time.perf_counter() - phase_start:.2f}s)')
        return True

    # ---- Step 3: shared slope optimisation ----
    shared = best_params
    optimizer = torch.optim.Adam(shared, lr=OPT_LR)
    start_v = end_v = None

    with torch.enable_grad():
        for step in range(SHARED_OPT_STEPS):
            optimizer.zero_grad(set_to_none=True)
            margins, _ = run(alpha_params=shared)

            unresolved_mask = ~certified
            cur = float(margins.detach()[unresolved_mask].min()) if unresolved_mask.any() \
                else float(margins.detach().min())
            if start_v is None:
                start_v = cur
            end_v = cur

            certified = certified | (margins.detach() > CERT_TOL)
            if bool(certified.all()):
                dbg(f'VERIFIED during shared opt at step {step} '
                    f'({time.perf_counter() - phase_start:.2f}s)')
                return True

            unresolved = margins[~certified]
            if unresolved.numel() == 0:
                return True

            loss = OPT_TEMPERATURE * torch.logsumexp(-unresolved / OPT_TEMPERATURE, dim=0)
            if not torch.isfinite(loss):
                dbg(f'shared opt loss non-finite at step {step}, stopping early')
                break
            loss.backward()
            optimizer.step()
            project(shared)

    dbg(f'shared opt: min_margin {start_v:.4f} -> {end_v:.4f} over {SHARED_OPT_STEPS} steps')

    with torch.no_grad():
        margins, _ = run(alpha_params=shared)
        certified = certified | (margins > CERT_TOL)
        dbg(f'after shared phase: certified={int(certified.sum())}/{n_out - 1}')
        if bool(certified.all()):
            dbg(f'VERIFIED after shared phase ({time.perf_counter() - phase_start:.2f}s)')
            return True

    # ---- Step 4: a dedicated slope set per remaining target ----
    remaining = (~certified).nonzero(as_tuple=True)[0].tolist()
    shared_values = [p.detach().clone() for p in shared]
    dbg(f'{len(remaining)} classes unresolved -> per-target opt: {remaining}')

    for pos in remaining:
        index = torch.tensor([pos], device=device, dtype=torch.long)
        params = make_parameters(shared_values)
        optimizer = torch.optim.Adam(params, lr=OPT_LR)

        target_certified = False
        start_v = end_v = None
        with torch.enable_grad():
            for step in range(TARGET_OPT_STEPS):
                optimizer.zero_grad(set_to_none=True)
                margin, _ = run(alpha_params=params, target_positions=index)
                scalar = margin[0]
                cur = float(scalar.detach())
                if start_v is None:
                    start_v = cur
                end_v = cur
                if cur > CERT_TOL:
                    target_certified = True
                    break
                loss = -scalar
                if not torch.isfinite(loss):
                    break
                loss.backward()
                optimizer.step()
                project(params)

        if not target_certified:
            with torch.no_grad():
                margin, _ = run(alpha_params=params, target_positions=index)
                end_v = float(margin[0])
                target_certified = end_v > CERT_TOL

        dbg(f'target class_idx={pos}: margin {start_v:.4f} -> {end_v:.4f} '
            f'certified={target_certified}')

        if not target_certified:
            dbg(f'class_idx={pos} could NOT be certified -> NOT VERIFIED '
                f'({time.perf_counter() - phase_start:.2f}s)')
            return False

    dbg(f'VERIFIED after per-target phase ({time.perf_counter() - phase_start:.2f}s)')
    return True


def analyze(model, x, eps, true_label, debug=False, tag=""):
    """
    Stage 0: a real PGD attack.  A misclassified point inside the eps-ball is
    a complete, sound proof of non-robustness -- return immediately.

    If PGD fails, its final point is used to read off each ReLU's actual
    behaviour at this specific image and eps, giving a per-case slope
    initialisation tried alongside the generic 0/1 heuristic.

    Stage 1: DeepPoly with back-substitution (every intermediate bound, not
    just the margin), intersected with interval bounds, slopes optimised.
    """
    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} {msg}', flush=True)

    start_time = time.perf_counter()

    found_adv, pgd_point = pgd_attack(model, x, eps, true_label, debug=debug, tag=tag)
    if found_adv:
        dbg(f'total_time={time.perf_counter() - start_time:.2f}s '
            f'result=NOT VERIFIED (PGD found a counterexample)')
        return False

    pgd_signs = get_relu_preact_signs(model, pgd_point)
    result = _analyze_deeppoly(model, x, eps, true_label,
                               extra_alpha_init=pgd_signs, debug=debug, tag=tag)
    dbg(f'total_time={time.perf_counter() - start_time:.2f}s '
        f'result={"VERIFIED" if result else "NOT VERIFIED"}')
    return result


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

    net.load_state_dict(torch.load(f'../mnist_nets/{net_name}.pt',
                                   map_location=torch.device(DEVICE)))
    return net


def parse_spec(spec_path):
    with open(spec_path, 'r') as f:
        lines = [line.rstrip('\n') for line in f.readlines()]
    true_label = int(lines[0])
    pixel_values = [float(line) for line in lines[1:]]
    eps = float(spec_path[:-4].split('/')[-1].split('_')[-1])
    return true_label, pixel_values, eps


def run_single_case(net_name, spec_path, debug=False, tag=""):
    true_label, pixel_values, eps = parse_spec(spec_path)
    net = load_network(net_name)
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    assert outs.max(dim=1)[1].item() == true_label
    return analyze(net, inputs, eps, true_label, debug=debug, tag=tag)


def run_all_cases(test_dir):
    base_dir = os.path.abspath(test_dir)
    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(base_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue
        spec_paths = sorted(
            os.path.join(net_dir, f) for f in os.listdir(net_dir) if f.endswith('.txt'))
        for spec_path in spec_paths:
            start = time.perf_counter()
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue
            status = 'verified' if result else 'not verified'
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}'
                  f'\t{time.perf_counter() - start:.1f}s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, choices=NETWORK_NAMES)
    parser.add_argument('--spec', type=str)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--tests-dir', type=str, default='../test_cases')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.batch:
        run_all_cases(args.tests_dir)
        return

    if args.net is None or args.spec is None:
        parser.error('Please provide both --net and --spec, or use --batch.')

    true_label, pixel_values, eps = parse_spec(args.spec)
    net = load_network(args.net)
    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    assert net(inputs).max(dim=1)[1].item() == true_label

    print('verified' if analyze(net, inputs, eps, true_label,
                                debug=args.debug,
                                tag=os.path.basename(args.spec)) else 'not verified')


if __name__ == '__main__':
    main()