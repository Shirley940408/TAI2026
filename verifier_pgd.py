import argparse
import os
import time

import torch
import torch.nn.functional as F
from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']


# =============================================================================
# PGD attack (input-space search for a real counterexample) and the helper
# that reads off ReLU pre-activation signs at a candidate point, used to build
# a per-image slope initialization for the verifier below.
# =============================================================================

def pgd_attack(model, x, eps, true_label, num_restarts=5, num_steps=50, step_size=None, debug=False, tag=""):
    """
    Standard L-inf PGD: random start inside the eps-ball, then sign-gradient
    ascent on cross-entropy loss, projected back into [x-eps, x+eps] and
    the valid pixel range [0, 1] after every step.

    Returns (found_adversarial: bool, representative_point: Tensor).
    If found_adversarial is True, representative_point is a real input that
    the network misclassifies -- this alone is a complete, sound proof of
    non-robustness (no abstraction needed at all). If False,
    representative_point is the endpoint of the last restart, used only as a
    heuristic signal for initializing ReLU slopes -- it carries no soundness
    weight by itself.
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
    """
    Run a clean forward pass through the network body at `point` and record,
    for every ReLU layer in order, whether each neuron's pre-activation was
    >= 0. Returns a list of {0.0, 1.0} tensors, one per ReLU layer, matching
    the same layer order used by the verifier below.
    """
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
            raise NotImplementedError('Unsupported layer type while reading PGD signs: {}'.format(type(layer)))
    return signs


# =============================================================================
# Stage 1 -- forward hybrid verifier (unchanged math from your version, plus
# an optional PGD-informed alpha candidate tried alongside the 0/1 heuristic).
# =============================================================================

def _analyze_forward_hybrid(model, x, eps, true_label, extra_alpha_init=None, debug=False, tag=""):
    SHARED_OPT_STEPS = 18
    TARGET_OPT_STEPS = 12
    OPT_LR = 5e-2
    OPT_TEMPERATURE = 0.5
    OPT_INIT_EPS = 5e-2
    CERT_TOL = 1e-8

    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} [forward] {msg}', flush=True)

    phase_start = time.perf_counter()

    model.eval()
    device = x.device
    dtype = x.dtype

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError('This implementation expects the final network layer to be Linear.')

    body_layers = layers[:-1]
    output_layer = layers[-1]

    x_detached = x.detach()
    x_flat = x_detached.reshape(-1)
    input_dim = x_flat.numel()

    original_x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    original_x_u = torch.clamp(x_flat + eps, 0.0, 1.0)
    original_box_l = original_x_l.reshape_as(x_detached)
    original_box_u = original_x_u.reshape_as(x_detached)

    other_labels = torch.tensor(
        [label for label in range(10) if label != true_label], device=device, dtype=torch.long,
    )

    float_eps = torch.finfo(dtype).eps

    def concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u):
        A_low_pos, A_low_neg = A_low.clamp(min=0), A_low.clamp(max=0)
        A_high_pos, A_high_neg = A_high.clamp(min=0), A_high.clamp(max=0)
        low = A_low_pos @ x_l + A_low_neg @ x_u + b_low
        high = A_high_pos @ x_u + A_high_neg @ x_l + b_high
        return low, high

    def concrete_lower(A, b, x_l, x_u):
        A_pos, A_neg = A.clamp(min=0), A.clamp(max=0)
        return A_pos @ x_l + A_neg @ x_u + b

    def make_parameters(alpha_values, move_inside=False):
        params = []
        for value in alpha_values:
            init = value.detach().clone()
            if move_inside:
                init.clamp_(OPT_INIT_EPS, 1.0 - OPT_INIT_EPS)
            params.append(torch.nn.Parameter(init))
        return params

    def project_alphas(alpha_params):
        with torch.no_grad():
            for alpha in alpha_params:
                alpha.clamp_(0.0, 1.0)

    def propagate(alpha_params=None, initialize_alpha=False, target_positions=None,
                  unstable_counts=None, l_u_record=None, pgd_alpha_params=None,
                  build_pgd_candidate=False):
        if alpha_params is None:
            alpha_params = []

        x_l = original_x_l
        x_u = original_x_u

        A_low = torch.eye(input_dim, device=device, dtype=dtype)
        A_high = torch.eye(input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(input_dim, device=device, dtype=dtype)

        box_low = original_box_l
        box_high = original_box_u

        cur_shape = tuple(x_detached.shape[-3:]) if x_detached.dim() >= 3 else None
        relu_id = 0

        for layer in body_layers:
            if isinstance(layer, Normalization):
                mean_full = layer.mean.detach().to(device=device, dtype=dtype)
                sigma_full = layer.sigma.detach().to(device=device, dtype=dtype)
                if torch.any(sigma_full <= 0):
                    raise ValueError('Normalization sigma must be positive.')
                box_low = (box_low - mean_full) / sigma_full
                box_high = (box_high - mean_full) / sigma_full
                mean_flat = mean_full.reshape(-1)
                sigma_flat = sigma_full.reshape(-1)
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
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low = new_b_low + b
                    new_b_high = new_b_high + b
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

                box_low_flat = box_low.reshape(box_low.size(0), -1)
                box_high_flat = box_high.reshape(box_high.size(0), -1)
                new_box_low = box_low_flat @ W_pos.t() + box_high_flat @ W_neg.t()
                new_box_high = box_high_flat @ W_pos.t() + box_low_flat @ W_neg.t()
                if b is not None:
                    new_box_low = new_box_low + b
                    new_box_high = new_box_high + b
                box_low, box_high = new_box_low, new_box_high
                cur_shape = None
                continue

            if isinstance(layer, torch.nn.Conv2d):
                if cur_shape is None:
                    if box_low.dim() != 4:
                        raise RuntimeError('Conv2d encountered without a valid feature-map shape.')
                    cur_shape = tuple(box_low.shape[-3:])

                W = layer.weight.detach().to(device=device, dtype=dtype)
                b = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                C_in, H_in, W_in = cur_shape
                W_pos, W_neg = W.clamp(min=0), W.clamp(max=0)

                def conv_coeff(weight, coeff):
                    coeff_img = coeff.t().reshape(input_dim, C_in, H_in, W_in)
                    return F.conv2d(coeff_img, weight, bias=None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups)

                low_img = conv_coeff(W_pos, A_low) + conv_coeff(W_neg, A_high)
                high_img = conv_coeff(W_pos, A_high) + conv_coeff(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                new_A_low = low_img.reshape(input_dim, -1).t()
                new_A_high = high_img.reshape(input_dim, -1).t()

                def conv_bias(weight, bvec):
                    b_img = bvec.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(b_img, weight, bias=None, stride=layer.stride,
                                     padding=layer.padding, dilation=layer.dilation, groups=layer.groups).reshape(-1)

                new_b_low = conv_bias(W_pos, b_low) + conv_bias(W_neg, b_high)
                new_b_high = conv_bias(W_pos, b_high) + conv_bias(W_neg, b_low)
                if b is not None:
                    expanded_bias = b.view(-1, 1, 1).expand(C_out, H_out, W_out).reshape(-1)
                    new_b_low = new_b_low + expanded_bias
                    new_b_high = new_b_high + expanded_bias
                A_low, A_high, b_low, b_high = new_A_low, new_A_high, new_b_low, new_b_high

                new_box_low = F.conv2d(box_low, W_pos, bias=None, stride=layer.stride, padding=layer.padding,
                                        dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(box_high, W_neg, bias=None, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                new_box_high = F.conv2d(box_high, W_pos, bias=None, stride=layer.stride, padding=layer.padding,
                                         dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(box_low, W_neg, bias=None, stride=layer.stride, padding=layer.padding,
                              dilation=layer.dilation, groups=layer.groups)
                if b is not None:
                    bias_img = b.view(1, -1, 1, 1)
                    new_box_low = new_box_low + bias_img
                    new_box_high = new_box_high + bias_img
                box_low, box_high = new_box_low, new_box_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.ReLU):
                affine_l, affine_u = concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u)
                box_l_flat = box_low.reshape(-1)
                box_u_flat = box_high.reshape(-1)

                l = torch.maximum(affine_l, box_l_flat)
                u = torch.minimum(affine_u, box_u_flat)
                u = torch.maximum(u, l)

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)
                neuron_count = l.numel()

                if unstable_counts is not None:
                    unstable_counts.append(int(cross_mask.sum().item()))
                if l_u_record is not None:
                    l_u_record.append((l.detach().clone(), u.detach().clone()))

                if initialize_alpha:
                    initial_alpha = torch.where(u >= -l, torch.ones_like(l), torch.zeros_like(l))
                    alpha_params.append(torch.nn.Parameter(initial_alpha.detach().clone()))

                if relu_id >= len(alpha_params):
                    raise RuntimeError('Missing alpha parameters for ReLU layer {}'.format(relu_id))
                alpha = alpha_params[relu_id]
                if alpha.numel() != neuron_count:
                    raise RuntimeError('Alpha size mismatch at ReLU layer {}'.format(relu_id))
                relu_id += 1

                alpha_safe = alpha.clamp(0.0, 1.0)
                zeros, ones = torch.zeros_like(l), torch.ones_like(l)

                lower_scale = torch.where(pos_mask, ones, torch.where(cross_mask, alpha_safe, zeros))
                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low

                denominator = torch.where(cross_mask, (u - l).clamp_min(float_eps), torch.ones_like(u))
                upper_slope_cross = torch.where(cross_mask, u / denominator, torch.zeros_like(u))
                upper_scale = torch.where(pos_mask, ones, torch.where(cross_mask, upper_slope_cross, zeros))
                upper_intercept = torch.where(cross_mask, -upper_slope_cross * l, zeros)
                A_high = upper_scale.unsqueeze(1) * A_high
                b_high = upper_scale * b_high + upper_intercept

                box_low = torch.clamp(l, min=0).reshape_as(box_low)
                box_high = torch.clamp(u, min=0).reshape_as(box_high)
                continue

            raise NotImplementedError('Unsupported layer type before output: {}'.format(type(layer)))

        W_out = output_layer.weight.detach().to(device=device, dtype=dtype)
        b_out = output_layer.bias
        b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype) if b_out is None else b_out.detach().to(device=device, dtype=dtype)

        selected_labels = other_labels if target_positions is None else other_labels.index_select(0, target_positions)

        W_margin = W_out[true_label].unsqueeze(0) - W_out.index_select(0, selected_labels)
        b_margin = b_out[true_label] - b_out.index_select(0, selected_labels)
        W_margin_pos, W_margin_neg = W_margin.clamp(min=0), W_margin.clamp(max=0)

        margin_A_low = W_margin_pos @ A_low + W_margin_neg @ A_high
        margin_b_low = W_margin_pos @ b_low + W_margin_neg @ b_high + b_margin
        affine_margin_lower = concrete_lower(margin_A_low, margin_b_low, x_l, x_u)

        box_l_flat = box_low.reshape(-1)
        box_u_flat = box_high.reshape(-1)
        box_margin_lower = W_margin_pos @ box_l_flat + W_margin_neg @ box_u_flat + b_margin

        hybrid_margin_lower = torch.maximum(affine_margin_lower, box_margin_lower)
        return hybrid_margin_lower, alpha_params

    # ---- Step 1: build the fixed 0/1 heuristic init, and record per-ReLU [l, u] ----
    initial_alpha_params = []
    l_u_record = []
    with torch.enable_grad():
        initial_margins, initial_alpha_params = propagate(
            alpha_params=initial_alpha_params, initialize_alpha=True, l_u_record=l_u_record,
        )

    dbg(f'0/1-heuristic min_margin={initial_margins.min().item():.4f} '
        f'already_certified={int((initial_margins.detach() > CERT_TOL).sum().item())}/9')

    certified = initial_margins.detach() > CERT_TOL
    if bool(certified.all()):
        dbg(f'VERIFIED from the 0/1 heuristic alone ({time.perf_counter() - phase_start:.2f}s)')
        return True

    if not initial_alpha_params:
        dbg(f'NO unstable ReLU neurons -> forward hybrid gives up ({time.perf_counter() - phase_start:.2f}s)')
        return False

    # ---- Step 2: if a PGD-informed candidate is available, build it and compare ----
    candidate_inits = [make_parameters(initial_alpha_params, move_inside=True)]
    candidate_names = ['heuristic']

    if extra_alpha_init is not None and len(extra_alpha_init) == len(l_u_record):
        pgd_alpha_full = []
        for pgd_sign, (l, u) in zip(extra_alpha_init, l_u_record):
            pos_mask = l >= 0
            cross_mask = (l < 0) & (u > 0)
            candidate = torch.where(
                pos_mask, torch.ones_like(l),
                torch.where(cross_mask, pgd_sign, torch.zeros_like(l)),
            )
            pgd_alpha_full.append(candidate)
        candidate_inits.append(make_parameters(pgd_alpha_full, move_inside=True))
        candidate_names.append('pgd-informed')

    best_params, best_score, best_margins, best_name = None, -float('inf'), None, None
    with torch.no_grad():
        for name, candidate in zip(candidate_names, candidate_inits):
            margins, _ = propagate(alpha_params=candidate)
            unresolved = margins[~certified]
            score = unresolved.min().item() if unresolved.numel() > 0 else margins.min().item()
            if score > best_score:
                best_score, best_params, best_margins, best_name = score, candidate, margins, name
    dbg(f'init comparison: chosen={best_name} score={best_score:.4f}')

    certified = certified | (best_margins > CERT_TOL)
    if bool(certified.all()):
        dbg(f'VERIFIED right after picking the better init ({time.perf_counter() - phase_start:.2f}s)')
        return True

    # ---- Step 3: shared optimization from the chosen init ----
    shared_params = best_params
    shared_optimizer = torch.optim.Adam(shared_params, lr=OPT_LR)

    shared_start, shared_end = None, None
    with torch.enable_grad():
        for step in range(SHARED_OPT_STEPS):
            shared_optimizer.zero_grad(set_to_none=True)
            margins, _ = propagate(alpha_params=shared_params)

            unresolved = ~certified
            cur_min = margins.detach()[unresolved].min().item() if unresolved.any() else margins.detach().min().item()
            if shared_start is None:
                shared_start = cur_min
            shared_end = cur_min

            certified = certified | (margins.detach() > CERT_TOL)
            if bool(certified.all()):
                dbg(f'VERIFIED during shared opt at step {step} ({time.perf_counter() - phase_start:.2f}s)')
                return True

            unresolved_margins = margins[~certified]
            if unresolved_margins.numel() == 0:
                return True

            loss = OPT_TEMPERATURE * torch.logsumexp(-unresolved_margins / OPT_TEMPERATURE, dim=0)
            if not torch.isfinite(loss):
                dbg(f'shared opt loss non-finite at step {step}, stopping early')
                break

            loss.backward()
            shared_optimizer.step()
            project_alphas(shared_params)

    dbg(f'shared opt: min_margin {shared_start:.4f} -> {shared_end:.4f} over {SHARED_OPT_STEPS} steps')

    with torch.no_grad():
        shared_margins, _ = propagate(alpha_params=shared_params)
        certified = certified | (shared_margins > CERT_TOL)
        dbg(f'after shared phase: certified={int(certified.sum().item())}/9')
        if bool(certified.all()):
            dbg(f'VERIFIED after shared phase ({time.perf_counter() - phase_start:.2f}s)')
            return True

    unresolved_positions = (~certified).nonzero(as_tuple=True)[0].tolist()
    shared_values = [alpha.detach().clone() for alpha in shared_params]
    dbg(f'{len(unresolved_positions)} classes unresolved -> per-target opt: {unresolved_positions}')

    for target_position in unresolved_positions:
        target_index = torch.tensor([target_position], device=device, dtype=torch.long)
        target_params = make_parameters(shared_values, move_inside=False)
        target_optimizer = torch.optim.Adam(target_params, lr=OPT_LR)

        target_certified = False
        start_v, end_v = None, None
        with torch.enable_grad():
            for step in range(TARGET_OPT_STEPS):
                target_optimizer.zero_grad(set_to_none=True)
                target_margin, _ = propagate(alpha_params=target_params, target_positions=target_index)
                scalar_margin = target_margin[0]
                cur_v = float(scalar_margin.detach())
                if start_v is None:
                    start_v = cur_v
                end_v = cur_v

                if cur_v > CERT_TOL:
                    target_certified = True
                    break

                loss = -scalar_margin
                if not torch.isfinite(loss):
                    break

                loss.backward()
                target_optimizer.step()
                project_alphas(target_params)

        if not target_certified:
            with torch.no_grad():
                final_target_margin, _ = propagate(alpha_params=target_params, target_positions=target_index)
                end_v = final_target_margin[0].item()
                target_certified = bool(end_v > CERT_TOL)

        dbg(f'target class_idx={target_position}: margin {start_v:.4f} -> {end_v:.4f} certified={target_certified}')

        if not target_certified:
            dbg(f'class_idx={target_position} could NOT be certified -> forward hybrid fails ({time.perf_counter() - phase_start:.2f}s)')
            return False

    dbg(f'VERIFIED after per-target phase ({time.perf_counter() - phase_start:.2f}s)')
    return True


# =============================================================================
# Stage 2 -- CROWN backward fallback (conv only), unchanged math, plus the
# same PGD-informed candidate added to its candidate list.
# =============================================================================

def _analyze_crown_backward(model, x, eps, true_label, time_budget_seconds=165.0, extra_alpha_init=None, debug=False, tag=""):
    CROWN_OPT_STEPS = 32
    CROWN_LR = 8e-2
    CERT_TOL = 1e-7
    PARAM_EPS = 2e-3

    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} [crown] {msg}', flush=True)

    phase_start = time.perf_counter()
    dbg(f'starting, time_budget_seconds={time_budget_seconds:.1f}')

    model.eval()
    device = x.device
    dtype = x.dtype
    deadline = time.perf_counter() + max(1.0, float(time_budget_seconds))

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError('CROWN fallback expects the final network layer to be Linear.')

    input_low = torch.clamp(x.detach() - eps, 0.0, 1.0)
    input_high = torch.clamp(x.detach() + eps, 0.0, 1.0)

    def _pair(value):
        return value if isinstance(value, tuple) else (value, value)

    def _sum_nonbatch(tensor):
        return tensor if tensor.dim() <= 1 else tensor.reshape(tensor.size(0), -1).sum(dim=1)

    records = []
    low, high = input_low, input_high

    with torch.no_grad():
        for layer in layers:
            input_shape = tuple(low.shape)

            if isinstance(layer, Normalization):
                mean = layer.mean.detach().to(device=device, dtype=dtype)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype)
                if torch.any(sigma <= 0):
                    raise ValueError('Normalization sigma must be positive.')
                low, high = (low - mean) / sigma, (high - mean) / sigma
                records.append({'layer': layer, 'input_shape': input_shape, 'output_shape': tuple(low.shape), 'mean': mean, 'sigma': sigma})
                continue

            if isinstance(layer, torch.nn.Flatten):
                low, high = low.reshape(low.size(0), -1), high.reshape(high.size(0), -1)
                records.append({'layer': layer, 'input_shape': input_shape, 'output_shape': tuple(low.shape)})
                continue

            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach().to(device=device, dtype=dtype)
                bias = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                weight_pos, weight_neg = weight.clamp(min=0), weight.clamp(max=0)
                old_low, old_high = low.reshape(low.size(0), -1), high.reshape(high.size(0), -1)
                low = old_low @ weight_pos.t() + old_high @ weight_neg.t()
                high = old_high @ weight_pos.t() + old_low @ weight_neg.t()
                if bias is not None:
                    low, high = low + bias, high + bias
                records.append({'layer': layer, 'input_shape': input_shape, 'output_shape': tuple(low.shape), 'weight': weight, 'bias': bias})
                continue

            if isinstance(layer, torch.nn.Conv2d):
                weight = layer.weight.detach().to(device=device, dtype=dtype)
                bias = None if layer.bias is None else layer.bias.detach().to(device=device, dtype=dtype)
                weight_pos, weight_neg = weight.clamp(min=0), weight.clamp(max=0)
                old_low, old_high = low, high
                low = F.conv2d(old_low, weight_pos, bias=None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(old_high, weight_neg, bias=None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
                high = F.conv2d(old_high, weight_pos, bias=None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups) + \
                    F.conv2d(old_low, weight_neg, bias=None, stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
                if bias is not None:
                    bias_img = bias.view(1, -1, 1, 1)
                    low, high = low + bias_img, high + bias_img
                records.append({'layer': layer, 'input_shape': input_shape, 'output_shape': tuple(low.shape), 'weight': weight, 'bias': bias})
                continue

            if isinstance(layer, torch.nn.ReLU):
                pre_low, pre_high = low.clone(), high.clone()
                low, high = torch.clamp(pre_low, min=0), torch.clamp(pre_high, min=0)
                records.append({'layer': layer, 'input_shape': input_shape, 'output_shape': tuple(low.shape), 'pre_low': pre_low, 'pre_high': pre_high})
                continue

            raise NotImplementedError('Unsupported layer type in CROWN forward pass: {}'.format(type(layer)))

    output_low, output_high = low.reshape(-1), high.reshape(-1)
    output_size = output_low.numel()
    if not (0 <= true_label < output_size):
        raise ValueError('true_label is outside the output dimension.')

    relu_records = [record for record in records if isinstance(record['layer'], torch.nn.ReLU)]

    def alpha_candidate(kind):
        values = []
        for idx, record in enumerate(relu_records):
            l, u = record['pre_low'].squeeze(0), record['pre_high'].squeeze(0)
            neg, pos = u <= 0, l >= 0
            unstable = (~neg) & (~pos)
            denominator = (u - l).clamp_min(torch.finfo(dtype).eps)
            upper_slope = torch.where(unstable, u / denominator, torch.zeros_like(u))

            if kind == 'heuristic':
                unstable_value = torch.where(u >= -l, torch.ones_like(u), torch.zeros_like(u))
            elif kind == 'half':
                unstable_value = torch.full_like(u, 0.5)
            elif kind == 'secant':
                unstable_value = upper_slope
            elif kind == 'pgd':
                pgd_sign = extra_alpha_init[idx].reshape_as(u)
                unstable_value = pgd_sign
            else:
                raise ValueError('Unknown alpha candidate: {}'.format(kind))

            alpha = torch.where(pos, torch.ones_like(u), torch.where(unstable, unstable_value, torch.zeros_like(u)))
            values.append(alpha)
        return values

    def theta_from_alpha(alpha_values):
        params = []
        for alpha in alpha_values:
            safe_alpha = alpha.detach().clamp(PARAM_EPS, 1.0 - PARAM_EPS)
            theta = torch.log(safe_alpha) - torch.log1p(-safe_alpha)
            params.append(torch.nn.Parameter(theta))
        return params

    def alpha_from_theta(theta_params):
        return [torch.sigmoid(theta) for theta in theta_params]

    def crown_margin_lower(target_label, alpha_values):
        coefficient = torch.zeros(1, output_size, device=device, dtype=dtype)
        coefficient[0, true_label] = 1.0
        coefficient[0, target_label] = -1.0
        bound_bias = torch.zeros(1, device=device, dtype=dtype)
        relu_index = len(relu_records) - 1

        for record in reversed(records):
            layer = record['layer']

            if isinstance(layer, torch.nn.Linear):
                weight, bias = record['weight'], record['bias']
                coefficient = coefficient.reshape(coefficient.size(0), -1)
                if bias is not None:
                    bound_bias = bound_bias + coefficient @ bias
                coefficient = coefficient @ weight
                continue

            if isinstance(layer, torch.nn.Flatten):
                coefficient = coefficient.reshape((coefficient.size(0),) + record['input_shape'][1:])
                continue

            if isinstance(layer, torch.nn.ReLU):
                l, u = record['pre_low'].squeeze(0), record['pre_high'].squeeze(0)
                alpha = alpha_values[relu_index]
                relu_index -= 1

                neg, pos = u <= 0, l >= 0
                unstable = (~neg) & (~pos)
                zeros, ones = torch.zeros_like(u), torch.ones_like(u)

                lower_slope = torch.where(pos, ones, torch.where(unstable, alpha, zeros))
                denominator = torch.where(unstable, (u - l).clamp_min(torch.finfo(dtype).eps), torch.ones_like(u))
                unstable_upper_slope = torch.where(unstable, u / denominator, zeros)
                upper_slope = torch.where(pos, ones, torch.where(unstable, unstable_upper_slope, zeros))
                upper_intercept = torch.where(unstable, -l * unstable_upper_slope, zeros)

                positive_coefficient, negative_coefficient = coefficient.clamp(min=0), coefficient.clamp(max=0)
                bound_bias = bound_bias + _sum_nonbatch(negative_coefficient * upper_intercept.unsqueeze(0))
                coefficient = positive_coefficient * lower_slope.unsqueeze(0) + negative_coefficient * upper_slope.unsqueeze(0)
                continue

            if isinstance(layer, torch.nn.Conv2d):
                weight, bias = record['weight'], record['bias']
                if coefficient.dim() != 4:
                    raise RuntimeError('Conv2d backward coefficient must be four-dimensional.')
                if bias is not None:
                    bound_bias = bound_bias + _sum_nonbatch(coefficient * bias.view(1, -1, 1, 1))

                stride_h, stride_w = _pair(layer.stride)
                pad_h, pad_w = _pair(layer.padding)
                dil_h, dil_w = _pair(layer.dilation)
                kernel_h, kernel_w = _pair(layer.kernel_size)
                input_h, input_w = record['input_shape'][-2:]
                output_h, output_w = coefficient.shape[-2:]

                base_h = (output_h - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + 1
                base_w = (output_w - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + 1
                output_padding_h, output_padding_w = input_h - base_h, input_w - base_w
                if not (0 <= output_padding_h < stride_h) or not (0 <= output_padding_w < stride_w):
                    raise RuntimeError('Invalid conv_transpose2d output_padding.')

                coefficient = F.conv_transpose2d(
                    coefficient, weight, bias=None, stride=layer.stride, padding=layer.padding,
                    output_padding=(output_padding_h, output_padding_w), groups=layer.groups, dilation=layer.dilation,
                )
                continue

            if isinstance(layer, Normalization):
                mean, sigma = record['mean'], record['sigma']
                bound_bias = bound_bias + _sum_nonbatch(coefficient * (-mean / sigma))
                coefficient = coefficient / sigma
                continue

            raise NotImplementedError('Unsupported layer type in CROWN backward pass: {}'.format(type(layer)))

        if relu_index != -1:
            raise RuntimeError('Not all ReLU slope tensors were consumed.')

        coefficient_pos, coefficient_neg = coefficient.clamp(min=0), coefficient.clamp(max=0)
        lower = bound_bias + _sum_nonbatch(coefficient_pos * input_low + coefficient_neg * input_high)
        return lower[0]

    target_labels = [label for label in range(output_size) if label != true_label]
    target_labels.sort(key=lambda label: float(output_low[true_label] - output_high[label]))

    candidate_names = ['heuristic', 'half', 'secant']
    if extra_alpha_init is not None and len(extra_alpha_init) == len(relu_records):
        candidate_names.append('pgd')
    candidate_sets = [alpha_candidate(name) for name in candidate_names]

    for target_label in target_labels:
        if time.perf_counter() >= deadline:
            dbg(f'deadline reached before target_label={target_label} -> NOT VERIFIED')
            return False

        ibp_margin = output_low[true_label] - output_high[target_label]
        if float(ibp_margin) > CERT_TOL:
            continue

        best_margin_value, best_alpha_values = -float('inf'), None
        with torch.no_grad():
            for candidate in candidate_sets:
                margin = crown_margin_lower(target_label, candidate)
                value = float(margin)
                if value > best_margin_value:
                    best_margin_value = value
                    best_alpha_values = [a.detach().clone() for a in candidate]
                if value > CERT_TOL:
                    break

        if best_margin_value > CERT_TOL:
            continue
        if best_alpha_values is None or not relu_records:
            return False

        theta_params = theta_from_alpha(best_alpha_values)
        optimizer = torch.optim.Adam(theta_params, lr=CROWN_LR)
        best_theta_state = [theta.detach().clone() for theta in theta_params]

        with torch.enable_grad():
            for step in range(CROWN_OPT_STEPS):
                if time.perf_counter() >= deadline:
                    break
                optimizer.zero_grad(set_to_none=True)
                current_alphas = alpha_from_theta(theta_params)
                margin = crown_margin_lower(target_label, current_alphas)
                margin_value = float(margin.detach())

                if margin_value > best_margin_value:
                    best_margin_value = margin_value
                    best_theta_state = [theta.detach().clone() for theta in theta_params]
                if margin_value > CERT_TOL:
                    break
                if not torch.isfinite(margin):
                    break

                (-margin).backward()
                optimizer.step()
                if step == CROWN_OPT_STEPS // 2:
                    for group in optimizer.param_groups:
                        group['lr'] *= 0.35

        if best_margin_value <= CERT_TOL:
            with torch.no_grad():
                restored_alphas = [torch.sigmoid(theta) for theta in best_theta_state]
                restored_margin = crown_margin_lower(target_label, restored_alphas)
                best_margin_value = max(best_margin_value, float(restored_margin))

        dbg(f'target_label={target_label}: best_margin={best_margin_value:.4f} certified={best_margin_value > CERT_TOL}')

        if best_margin_value <= CERT_TOL:
            dbg(f'target_label={target_label} could NOT be certified -> NOT VERIFIED ({time.perf_counter() - phase_start:.2f}s)')
            return False

    dbg(f'VERIFIED, all target labels certified ({time.perf_counter() - phase_start:.2f}s)')
    return True


def analyze(model, x, eps, true_label, debug=False, tag=""):
    """
    New Stage 0: run a real PGD attack in input space first. If it finds an
    actual misclassified point inside the eps-ball, that's a complete, sound
    proof of non-robustness -- return immediately, no abstraction needed.

    If PGD fails to find one, its final (failed) point is used to read off
    each ReLU's real behavior at that specific image and eps, giving a
    per-case slope initialization that's tried alongside the generic 0/1
    heuristic in both Stage 1 (forward hybrid) and Stage 2 (CROWN backward).
    """
    def dbg(msg):
        if debug:
            print(f'[dbg]{(" " + tag) if tag else ""} {msg}', flush=True)

    start_time = time.perf_counter()

    found_adv, pgd_point = pgd_attack(model, x, eps, true_label, debug=debug, tag=tag)
    if found_adv:
        dbg(f'total_time={time.perf_counter() - start_time:.2f}s result=NOT VERIFIED (PGD found a counterexample)')
        return False

    pgd_signs = get_relu_preact_signs(model, pgd_point)

    forward_result = _analyze_forward_hybrid(model, x, eps, true_label, extra_alpha_init=pgd_signs, debug=debug, tag=tag)
    if forward_result:
        dbg(f'total_time={time.perf_counter() - start_time:.2f}s result=VERIFIED (forward)')
        return True

    has_convolution = any(isinstance(layer, torch.nn.Conv2d) for layer in model.layers.children())
    if not has_convolution:
        dbg(f'total_time={time.perf_counter() - start_time:.2f}s result=NOT VERIFIED (no conv, no fallback)')
        return False

    elapsed = time.perf_counter() - start_time
    remaining_budget = 168.0 - elapsed
    if remaining_budget < 5.0:
        dbg(f'total_time={time.perf_counter() - start_time:.2f}s result=NOT VERIFIED (not enough budget left for fallback)')
        return False

    result = _analyze_crown_backward(
        model, x, eps, true_label, time_budget_seconds=remaining_budget,
        extra_alpha_init=pgd_signs, debug=debug, tag=tag,
    )
    dbg(f'total_time={time.perf_counter() - start_time:.2f}s result={"VERIFIED" if result else "NOT VERIFIED"} (crown fallback)')
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

    net.load_state_dict(torch.load(f'../mnist_nets/{net_name}.pt', map_location=torch.device(DEVICE)))
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
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label
    return analyze(net, inputs, eps, true_label, debug=debug, tag=tag)


def run_all_cases(test_dir):
    base_dir = os.path.abspath(test_dir)
    for net_name in NETWORK_NAMES:
        net_dir = os.path.join(base_dir, net_name)
        if not os.path.isdir(net_dir):
            print(f'[skip] {net_name}: missing directory {net_dir}')
            continue
        spec_paths = sorted(
            os.path.join(net_dir, filename) for filename in os.listdir(net_dir) if filename.endswith('.txt')
        )
        for spec_path in spec_paths:
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue
            status = 'verified' if result else 'not verified'
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}')


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
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    print('verified' if analyze(net, inputs, eps, true_label, debug=args.debug, tag=os.path.basename(args.spec)) else 'not verified')


if __name__ == '__main__':
    main()