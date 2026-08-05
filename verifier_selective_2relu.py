import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from networks import FullyConnected, Conv, Normalization

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']


def _analyze_forward_hybrid(model, x, eps, true_label):
    """
    Hybrid IBP + affine-bound verifier with optimized ReLU slopes.

    Main changes compared with the previous version:
      1. Keep both symbolic affine bounds and concrete IBP bounds.
      2. At every ReLU, intersect the two pre-activation intervals before
         constructing the relaxation. This keeps the useful fact ReLU(z) >= 0.
      3. Replace separate output-logit bounds with direct pairwise margins:
             f_true(x) - f_other(x) > 0.
      4. First optimize one shared set of slopes; then optimize a separate set
         for each still-unproved target class. Different sound relaxations may
         be used to prove different pairwise inequalities.

    For every unstable ReLU l < 0 < u, the symbolic lower relaxation is

        ReLU(z) >= alpha * z,   alpha in [0, 1].

    All computation is CPU-compatible and uses only PyTorch.
    """
    # Keep these modest for the 3-minute per-case limit. Increase them only
    # after measuring the slowest convolutional case on your machine.
    SHARED_OPT_STEPS = 18
    TARGET_OPT_STEPS = 12
    OPT_LR = 5e-2
    OPT_TEMPERATURE = 0.5
    OPT_INIT_EPS = 5e-2
    CERT_TOL = 1e-8

    model.eval()
    device = x.device
    dtype = x.dtype

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError(
            'This implementation expects the final network layer to be Linear.'
        )

    body_layers = layers[:-1]
    output_layer = layers[-1]

    x_detached = x.detach()
    x_flat = x_detached.reshape(-1)
    input_dim = x_flat.numel()

    # The project perturbation set is also clipped to the valid image range.
    original_x_l = torch.clamp(x_flat - eps, 0.0, 1.0)
    original_x_u = torch.clamp(x_flat + eps, 0.0, 1.0)
    original_box_l = original_x_l.reshape_as(x_detached)
    original_box_u = original_x_u.reshape_as(x_detached)

    other_labels = torch.tensor(
        [label for label in range(10) if label != true_label],
        device=device,
        dtype=torch.long,
    )

    float_eps = torch.finfo(dtype).eps

    def concrete_bounds(A_low, b_low, A_high, b_high, x_l, x_u):
        """Concretize affine lower/upper expressions over [x_l, x_u]."""
        A_low_pos = A_low.clamp(min=0)
        A_low_neg = A_low.clamp(max=0)
        A_high_pos = A_high.clamp(min=0)
        A_high_neg = A_high.clamp(max=0)

        low = A_low_pos @ x_l + A_low_neg @ x_u + b_low
        high = A_high_pos @ x_u + A_high_neg @ x_l + b_high
        return low, high

    def concrete_lower(A, b, x_l, x_u):
        """Concretize only a symbolic lower affine expression."""
        A_pos = A.clamp(min=0)
        A_neg = A.clamp(max=0)
        return A_pos @ x_l + A_neg @ x_u + b

    def make_parameters(alpha_values, move_inside=False):
        """Clone slope tensors as independent learnable parameters."""
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

    def propagate(alpha_params=None, initialize_alpha=False, target_positions=None):
        """
        Propagate hybrid bounds and return direct lower bounds on output margins.

        target_positions indexes `other_labels`, not the original class labels.
        When it is None, all nine pairwise margins are returned.
        """
        if alpha_params is None:
            alpha_params = []

        # Symbolic bounds are expressed over the current normalized input box.
        x_l = original_x_l
        x_u = original_x_u

        A_low = torch.eye(input_dim, device=device, dtype=dtype)
        A_high = torch.eye(input_dim, device=device, dtype=dtype)
        b_low = torch.zeros(input_dim, device=device, dtype=dtype)
        b_high = torch.zeros(input_dim, device=device, dtype=dtype)

        # A second, independent IBP state is preserved throughout the network.
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

                # Update the IBP tensor state.
                box_low = (box_low - mean_full) / sigma_full
                box_high = (box_high - mean_full) / sigma_full

                # The symbolic identity is now interpreted over the normalized
                # input interval, so only its domain endpoints change.
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
                b = None if layer.bias is None else layer.bias.detach().to(
                    device=device, dtype=dtype
                )
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                # Symbolic affine propagation.
                new_A_low = W_pos @ A_low + W_neg @ A_high
                new_A_high = W_pos @ A_high + W_neg @ A_low
                new_b_low = W_pos @ b_low + W_neg @ b_high
                new_b_high = W_pos @ b_high + W_neg @ b_low
                if b is not None:
                    new_b_low = new_b_low + b
                    new_b_high = new_b_high + b
                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high

                # Parallel IBP propagation.
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
                    # Recover the feature-map shape directly from the IBP state.
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
                W_pos = W.clamp(min=0)
                W_neg = W.clamp(max=0)

                def conv_coeff(weight, coeff):
                    # coeff: [N_previous, input_dim]. Treat input_dim as batch.
                    coeff_img = coeff.t().reshape(input_dim, C_in, H_in, W_in)
                    return F.conv2d(
                        coeff_img,
                        weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )

                low_img = conv_coeff(W_pos, A_low) + conv_coeff(W_neg, A_high)
                high_img = conv_coeff(W_pos, A_high) + conv_coeff(W_neg, A_low)
                C_out, H_out, W_out = low_img.shape[1:]
                new_A_low = low_img.reshape(input_dim, -1).t()
                new_A_high = high_img.reshape(input_dim, -1).t()

                def conv_bias(weight, bias_vector):
                    bias_img = bias_vector.reshape(1, C_in, H_in, W_in)
                    return F.conv2d(
                        bias_img,
                        weight,
                        bias=None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    ).reshape(-1)

                new_b_low = conv_bias(W_pos, b_low) + conv_bias(W_neg, b_high)
                new_b_high = conv_bias(W_pos, b_high) + conv_bias(W_neg, b_low)
                if b is not None:
                    expanded_bias = b.view(-1, 1, 1).expand(
                        C_out, H_out, W_out
                    ).reshape(-1)
                    new_b_low = new_b_low + expanded_bias
                    new_b_high = new_b_high + expanded_bias
                A_low, A_high = new_A_low, new_A_high
                b_low, b_high = new_b_low, new_b_high

                # Parallel convolutional IBP propagation.
                new_box_low = F.conv2d(
                    box_low,
                    W_pos,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                ) + F.conv2d(
                    box_high,
                    W_neg,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                new_box_high = F.conv2d(
                    box_high,
                    W_pos,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                ) + F.conv2d(
                    box_low,
                    W_neg,
                    bias=None,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                )
                if b is not None:
                    bias_img = b.view(1, -1, 1, 1)
                    new_box_low = new_box_low + bias_img
                    new_box_high = new_box_high + bias_img
                box_low, box_high = new_box_low, new_box_high
                cur_shape = (C_out, H_out, W_out)
                continue

            if isinstance(layer, torch.nn.ReLU):
                affine_l, affine_u = concrete_bounds(
                    A_low, b_low, A_high, b_high, x_l, x_u
                )
                box_l_flat = box_low.reshape(-1)
                box_u_flat = box_high.reshape(-1)

                # Intersect two independently sound abstractions.
                l = torch.maximum(affine_l, box_l_flat)
                u = torch.minimum(affine_u, box_u_flat)

                # Avoid tiny contradictory intervals caused only by float noise.
                u = torch.maximum(u, l)

                neg_mask = u <= 0
                pos_mask = l >= 0
                cross_mask = (~neg_mask) & (~pos_mask)
                neuron_count = l.numel()

                if initialize_alpha:
                    initial_alpha = torch.where(
                        u >= -l,
                        torch.ones_like(l),
                        torch.zeros_like(l),
                    )
                    alpha_params.append(
                        torch.nn.Parameter(initial_alpha.detach().clone())
                    )

                if relu_id >= len(alpha_params):
                    raise RuntimeError(
                        'Missing alpha parameters for ReLU layer {}'.format(relu_id)
                    )
                alpha = alpha_params[relu_id]
                if alpha.numel() != neuron_count:
                    raise RuntimeError(
                        'Alpha size mismatch at ReLU layer {}: expected {}, got {}'.format(
                            relu_id, neuron_count, alpha.numel()
                        )
                    )
                relu_id += 1

                alpha_safe = alpha.clamp(0.0, 1.0)
                zeros = torch.zeros_like(l)
                ones = torch.ones_like(l)

                # Symbolic lower relaxation. The separate box state retains
                # the stronger concrete fact ReLU(z) >= 0.
                lower_scale = torch.where(
                    pos_mask,
                    ones,
                    torch.where(cross_mask, alpha_safe, zeros),
                )
                A_low = lower_scale.unsqueeze(1) * A_low
                b_low = lower_scale * b_low

                # Standard secant upper relaxation over the tightened [l, u].
                denominator = torch.where(
                    cross_mask,
                    (u - l).clamp_min(float_eps),
                    torch.ones_like(u),
                )
                upper_slope_cross = torch.where(
                    cross_mask,
                    u / denominator,
                    torch.zeros_like(u),
                )
                upper_scale = torch.where(
                    pos_mask,
                    ones,
                    torch.where(cross_mask, upper_slope_cross, zeros),
                )
                upper_intercept = torch.where(
                    cross_mask,
                    -upper_slope_cross * l,
                    zeros,
                )
                A_high = upper_scale.unsqueeze(1) * A_high
                b_high = upper_scale * b_high + upper_intercept

                # Keep exact interval non-negativity for all future IBP steps.
                box_low = torch.clamp(l, min=0).reshape_as(box_low)
                box_high = torch.clamp(u, min=0).reshape_as(box_high)
                continue

            raise NotImplementedError(
                'Unsupported layer type before output: {}'.format(type(layer))
            )

        # Directly propagate pairwise margins through the final Linear layer.
        W_out = output_layer.weight.detach().to(device=device, dtype=dtype)
        b_out = output_layer.bias
        if b_out is None:
            b_out = torch.zeros(W_out.size(0), device=device, dtype=dtype)
        else:
            b_out = b_out.detach().to(device=device, dtype=dtype)

        if target_positions is None:
            selected_labels = other_labels
        else:
            selected_labels = other_labels.index_select(0, target_positions)

        W_margin = W_out[true_label].unsqueeze(0) - W_out.index_select(
            0, selected_labels
        )
        b_margin = b_out[true_label] - b_out.index_select(0, selected_labels)
        W_margin_pos = W_margin.clamp(min=0)
        W_margin_neg = W_margin.clamp(max=0)

        margin_A_low = W_margin_pos @ A_low + W_margin_neg @ A_high
        margin_b_low = (
            W_margin_pos @ b_low
            + W_margin_neg @ b_high
            + b_margin
        )
        affine_margin_lower = concrete_lower(
            margin_A_low, margin_b_low, x_l, x_u
        )

        box_l_flat = box_low.reshape(-1)
        box_u_flat = box_high.reshape(-1)
        box_margin_lower = (
            W_margin_pos @ box_l_flat
            + W_margin_neg @ box_u_flat
            + b_margin
        )

        # Either sound abstraction may prove the margin; their maximum remains
        # a sound lower bound because both are lower bounds on the same value.
        hybrid_margin_lower = torch.maximum(
            affine_margin_lower,
            box_margin_lower,
        )
        return hybrid_margin_lower, alpha_params

    # Build the initial 0/1 DeepPoly heuristic and evaluate all margins.
    initial_alpha_params = []
    with torch.enable_grad():
        initial_margins, initial_alpha_params = propagate(
            alpha_params=initial_alpha_params,
            initialize_alpha=True,
        )

    certified = initial_margins.detach() > CERT_TOL
    if bool(certified.all()):
        return True

    if not initial_alpha_params:
        return False

    # Preserve the exact heuristic as a candidate, but optimize from slightly
    # inside the interval so projected Adam is less likely to stick at 0 or 1.
    shared_params = make_parameters(initial_alpha_params, move_inside=True)
    shared_optimizer = torch.optim.Adam(shared_params, lr=OPT_LR)

    with torch.enable_grad():
        for _ in range(SHARED_OPT_STEPS):
            shared_optimizer.zero_grad(set_to_none=True)
            margins, _ = propagate(alpha_params=shared_params)

            # Different iterations may certify different target classes. This
            # is sound: each pairwise inequality may use its own relaxation.
            certified = certified | (margins.detach() > CERT_TOL)
            if bool(certified.all()):
                return True

            unresolved = ~certified
            unresolved_margins = margins[unresolved]
            if unresolved_margins.numel() == 0:
                return True

            loss = OPT_TEMPERATURE * torch.logsumexp(
                -unresolved_margins / OPT_TEMPERATURE,
                dim=0,
            )
            if not torch.isfinite(loss):
                break

            loss.backward()
            shared_optimizer.step()
            project_alphas(shared_params)

    # Check the state after the final shared optimizer update.
    with torch.no_grad():
        shared_margins, _ = propagate(alpha_params=shared_params)
        certified = certified | (shared_margins > CERT_TOL)
        if bool(certified.all()):
            return True

    # Optimize a separate slope set for each remaining target class. A distinct
    # alpha proof for each f_true - f_target inequality is still fully sound.
    unresolved_positions = (~certified).nonzero(as_tuple=True)[0].tolist()

    # Start each target-specific search from the shared result, which usually
    # gives a better warm start than returning to the original 0/1 heuristic.
    shared_values = [alpha.detach().clone() for alpha in shared_params]

    for target_position in unresolved_positions:
        target_index = torch.tensor(
            [target_position], device=device, dtype=torch.long
        )
        target_params = make_parameters(shared_values, move_inside=False)
        target_optimizer = torch.optim.Adam(target_params, lr=OPT_LR)

        target_certified = False
        with torch.enable_grad():
            for _ in range(TARGET_OPT_STEPS):
                target_optimizer.zero_grad(set_to_none=True)
                target_margin, _ = propagate(
                    alpha_params=target_params,
                    target_positions=target_index,
                )
                scalar_margin = target_margin[0]

                if float(scalar_margin.detach()) > CERT_TOL:
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
                final_target_margin, _ = propagate(
                    alpha_params=target_params,
                    target_positions=target_index,
                )
                target_certified = bool(
                    final_target_margin[0].item() > CERT_TOL
                )

        if not target_certified:
            return False

    return True






def _analyze_selective_2relu_lp(
    model,
    x,
    eps,
    true_label,
    time_budget_seconds=150.0,
):
    """
    Whole-network sparse LP relaxation with selective 2-ReLU joint cuts.

    The base LP uses exact affine layer equations and the standard triangle
    relaxation for every unstable ReLU. For a small number of influential
    pairs in the first ReLU layer, it adds sound joint convex-hull cuts built
    from an outer polygon of the two pre-activations over the input box.

    The 2-ReLU path is a fallback only. It never returns verified unless every
    pairwise margin LP has a strictly positive optimum.
    """
    CERT_TOL = 1e-6
    MAX_PAIRS = 10
    MAX_CANDIDATES = 20
    SUPPORT_DIRECTIONS = 24
    SUPPORT_SLACK = 1e-9
    CUT_SLACK = 2e-7
    GEOM_TOL = 1e-8

    model.eval()
    device = x.device
    dtype = torch.float64
    deadline = time.perf_counter() + max(1.0, float(time_budget_seconds))

    layers = list(model.layers.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise NotImplementedError(
            'Selective 2-ReLU LP expects the final layer to be Linear.'
        )

    x64 = x.detach().to(device=device, dtype=dtype)
    raw_shape = tuple(x64.shape[1:])
    input_dim = x64.numel()
    input_l = torch.clamp(x64.reshape(-1) - eps, 0.0, 1.0)
    input_u = torch.clamp(x64.reshape(-1) + eps, 0.0, 1.0)

    # ------------------------------------------------------------------
    # 1) IBP bounds for every layer, plus the exact affine map from the raw
    #    input to the first ReLU pre-activation.
    # ------------------------------------------------------------------
    records = []
    low = input_l.reshape((1,) + raw_shape)
    high = input_u.reshape((1,) + raw_shape)

    exact_A = torch.eye(input_dim, dtype=dtype, device=device)
    exact_b = torch.zeros(input_dim, dtype=dtype, device=device)
    exact_shape = raw_shape
    tracking_first_affine = True
    first_relu_A = None
    first_relu_b = None
    first_relu_l = None
    first_relu_u = None

    def _pair(value):
        return value if isinstance(value, tuple) else (value, value)

    with torch.no_grad():
        for layer_index, layer in enumerate(layers):
            input_shape = tuple(low.shape)

            if isinstance(layer, Normalization):
                mean = layer.mean.detach().to(device=device, dtype=dtype)
                sigma = layer.sigma.detach().to(device=device, dtype=dtype)
                if torch.any(sigma <= 0):
                    raise ValueError('Normalization sigma must be positive.')

                low = (low - mean) / sigma
                high = (high - mean) / sigma

                if tracking_first_affine:
                    mean_flat = mean.reshape(-1)
                    sigma_flat = sigma.reshape(-1)
                    exact_A = exact_A / sigma_flat.unsqueeze(1)
                    exact_b = (exact_b - mean_flat) / sigma_flat

                records.append({
                    'layer': layer,
                    'kind': 'normalization',
                    'input_shape': input_shape,
                    'output_shape': tuple(low.shape),
                    'low': low.clone(),
                    'high': high.clone(),
                    'mean': mean,
                    'sigma': sigma,
                })
                continue

            if isinstance(layer, torch.nn.Flatten):
                low = low.reshape(low.size(0), -1)
                high = high.reshape(high.size(0), -1)
                if tracking_first_affine:
                    exact_shape = (exact_A.size(0),)
                records.append({
                    'layer': layer,
                    'kind': 'flatten',
                    'input_shape': input_shape,
                    'output_shape': tuple(low.shape),
                    'low': low.clone(),
                    'high': high.clone(),
                })
                continue

            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.detach().to(device=device, dtype=dtype)
                bias = (
                    torch.zeros(weight.size(0), dtype=dtype, device=device)
                    if layer.bias is None
                    else layer.bias.detach().to(device=device, dtype=dtype)
                )
                w_pos = weight.clamp(min=0)
                w_neg = weight.clamp(max=0)
                old_low = low.reshape(low.size(0), -1)
                old_high = high.reshape(high.size(0), -1)
                low = old_low @ w_pos.t() + old_high @ w_neg.t() + bias
                high = old_high @ w_pos.t() + old_low @ w_neg.t() + bias

                if tracking_first_affine:
                    exact_A = weight @ exact_A
                    exact_b = weight @ exact_b + bias
                    exact_shape = (weight.size(0),)

                records.append({
                    'layer': layer,
                    'kind': 'linear',
                    'input_shape': input_shape,
                    'output_shape': tuple(low.shape),
                    'low': low.clone(),
                    'high': high.clone(),
                    'weight': weight,
                    'bias': bias,
                })
                continue

            if isinstance(layer, torch.nn.Conv2d):
                weight = layer.weight.detach().to(device=device, dtype=dtype)
                bias = (
                    torch.zeros(weight.size(0), dtype=dtype, device=device)
                    if layer.bias is None
                    else layer.bias.detach().to(device=device, dtype=dtype)
                )
                w_pos = weight.clamp(min=0)
                w_neg = weight.clamp(max=0)
                old_low, old_high = low, high
                low = (
                    F.conv2d(
                        old_low, w_pos, None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                    + F.conv2d(
                        old_high, w_neg, None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                    + bias.view(1, -1, 1, 1)
                )
                high = (
                    F.conv2d(
                        old_high, w_pos, None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                    + F.conv2d(
                        old_low, w_neg, None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                    + bias.view(1, -1, 1, 1)
                )

                if tracking_first_affine:
                    c_in, h_in, w_in = exact_shape
                    coeff_img = exact_A.t().reshape(
                        input_dim, c_in, h_in, w_in
                    )
                    coeff_out = F.conv2d(
                        coeff_img,
                        weight,
                        None,
                        stride=layer.stride,
                        padding=layer.padding,
                        dilation=layer.dilation,
                        groups=layer.groups,
                    )
                    c_out, h_out, w_out = coeff_out.shape[1:]
                    exact_A = coeff_out.reshape(input_dim, -1).t()

                    bias_img = exact_b.reshape(1, c_in, h_in, w_in)
                    exact_b = (
                        F.conv2d(
                            bias_img,
                            weight,
                            None,
                            stride=layer.stride,
                            padding=layer.padding,
                            dilation=layer.dilation,
                            groups=layer.groups,
                        )
                        + bias.view(1, -1, 1, 1)
                    ).reshape(-1)
                    exact_shape = (c_out, h_out, w_out)

                records.append({
                    'layer': layer,
                    'kind': 'conv',
                    'input_shape': input_shape,
                    'output_shape': tuple(low.shape),
                    'low': low.clone(),
                    'high': high.clone(),
                    'weight': weight,
                    'bias': bias,
                })
                continue

            if isinstance(layer, torch.nn.ReLU):
                pre_low = low.clone()
                pre_high = high.clone()

                if first_relu_A is None:
                    first_relu_A = exact_A.detach().clone()
                    first_relu_b = exact_b.detach().clone()
                    first_relu_l = pre_low.reshape(-1).detach().clone()
                    first_relu_u = pre_high.reshape(-1).detach().clone()
                tracking_first_affine = False

                low = torch.clamp(pre_low, min=0)
                high = torch.clamp(pre_high, min=0)
                records.append({
                    'layer': layer,
                    'kind': 'relu',
                    'input_shape': input_shape,
                    'output_shape': tuple(low.shape),
                    'low': low.clone(),
                    'high': high.clone(),
                    'pre_low': pre_low,
                    'pre_high': pre_high,
                })
                continue

            raise NotImplementedError(
                'Unsupported layer type in 2-ReLU LP: {}'.format(type(layer))
            )

    if first_relu_A is None:
        return False

    output_low = low.reshape(-1)
    output_high = high.reshape(-1)
    output_size = output_low.numel()
    if output_size <= true_label:
        raise ValueError('true_label is outside the output dimension.')

    # ------------------------------------------------------------------
    # 2) Build one sparse LP for the whole network.
    # ------------------------------------------------------------------
    variable_bounds = [
        (float(input_l[i]), float(input_u[i])) for i in range(input_dim)
    ]
    variable_count = input_dim
    current_indices = np.arange(input_dim, dtype=np.int64).reshape(raw_shape)

    eq_row = []
    eq_col = []
    eq_data = []
    eq_rhs = []
    ub_row = []
    ub_col = []
    ub_data = []
    ub_rhs = []
    eq_count = 0
    ub_count = 0

    first_relu_pre_indices = None
    first_relu_post_indices = None

    def allocate(bounds_low, bounds_high, shape):
        nonlocal variable_count
        flat_low = np.asarray(bounds_low, dtype=np.float64).reshape(-1)
        flat_high = np.asarray(bounds_high, dtype=np.float64).reshape(-1)
        start = variable_count
        variable_count += flat_low.size
        for lo, hi in zip(flat_low, flat_high):
            if hi < lo:
                hi = lo
            variable_bounds.append((float(lo), float(hi)))
        return np.arange(start, variable_count, dtype=np.int64).reshape(shape)

    def add_eq(indices, values, rhs):
        nonlocal eq_count
        for index, value in zip(indices, values):
            if abs(value) > 0.0:
                eq_row.append(eq_count)
                eq_col.append(int(index))
                eq_data.append(float(value))
        eq_rhs.append(float(rhs))
        eq_count += 1

    def add_ub(indices, values, rhs):
        nonlocal ub_count
        for index, value in zip(indices, values):
            if abs(value) > 0.0:
                ub_row.append(ub_count)
                ub_col.append(int(index))
                ub_data.append(float(value))
        ub_rhs.append(float(rhs))
        ub_count += 1

    for record in records:
        kind = record['kind']
        layer = record['layer']
        out_shape = tuple(record['output_shape'][1:])
        out_low_np = record['low'].detach().cpu().numpy().reshape(out_shape)
        out_high_np = record['high'].detach().cpu().numpy().reshape(out_shape)

        if kind == 'flatten':
            current_indices = current_indices.reshape(out_shape)
            continue

        if kind == 'normalization':
            out_indices = allocate(out_low_np, out_high_np, out_shape)
            mean_raw = record['mean'].detach().cpu().numpy()
            sigma_raw = record['sigma'].detach().cpu().numpy()
            mean = np.broadcast_to(mean_raw, (1,) + out_shape).reshape(-1)
            sigma = np.broadcast_to(sigma_raw, (1,) + out_shape).reshape(-1)
            in_flat = current_indices.reshape(-1)
            out_flat = out_indices.reshape(-1)
            for k in range(out_flat.size):
                add_eq(
                    [out_flat[k], in_flat[k]],
                    [1.0, -1.0 / sigma[k]],
                    -mean[k] / sigma[k],
                )
            current_indices = out_indices
            continue

        if kind == 'linear':
            out_indices = allocate(out_low_np, out_high_np, out_shape)
            weight = record['weight'].detach().cpu().numpy()
            bias = record['bias'].detach().cpu().numpy()
            in_flat = current_indices.reshape(-1)
            out_flat = out_indices.reshape(-1)
            for out_i in range(out_flat.size):
                nz = np.flatnonzero(weight[out_i])
                indices = [out_flat[out_i]]
                values = [1.0]
                if nz.size:
                    indices.extend(in_flat[nz].tolist())
                    values.extend((-weight[out_i, nz]).tolist())
                add_eq(indices, values, bias[out_i])
            current_indices = out_indices
            continue

        if kind == 'conv':
            out_indices = allocate(out_low_np, out_high_np, out_shape)
            weight = record['weight'].detach().cpu().numpy()
            bias = record['bias'].detach().cpu().numpy()
            c_out, h_out, w_out = out_shape
            c_in, h_in, w_in = current_indices.shape
            k_h, k_w = weight.shape[-2:]
            stride_h, stride_w = _pair(layer.stride)
            pad_h, pad_w = _pair(layer.padding)
            dil_h, dil_w = _pair(layer.dilation)
            groups = int(layer.groups)
            in_per_group = c_in // groups
            out_per_group = c_out // groups

            for oc in range(c_out):
                group = oc // out_per_group
                ic_start = group * in_per_group
                for oh in range(h_out):
                    for ow in range(w_out):
                        indices = [out_indices[oc, oh, ow]]
                        values = [1.0]
                        for local_ic in range(in_per_group):
                            ic = ic_start + local_ic
                            for kh in range(k_h):
                                ih = oh * stride_h - pad_h + kh * dil_h
                                if ih < 0 or ih >= h_in:
                                    continue
                                for kw in range(k_w):
                                    iw = ow * stride_w - pad_w + kw * dil_w
                                    if iw < 0 or iw >= w_in:
                                        continue
                                    coeff = weight[oc, local_ic, kh, kw]
                                    if coeff != 0.0:
                                        indices.append(current_indices[ic, ih, iw])
                                        values.append(-coeff)
                        add_eq(indices, values, bias[oc])
            current_indices = out_indices
            continue

        if kind == 'relu':
            pre_indices = current_indices.copy()
            pre_l = record['pre_low'].detach().cpu().numpy().reshape(-1)
            pre_u = record['pre_high'].detach().cpu().numpy().reshape(-1)
            post_l = np.maximum(pre_l, 0.0)
            post_u = np.maximum(pre_u, 0.0)
            post_indices = allocate(post_l, post_u, out_shape)

            if first_relu_pre_indices is None:
                first_relu_pre_indices = pre_indices.reshape(-1).copy()
                first_relu_post_indices = post_indices.reshape(-1).copy()

            pre_flat = pre_indices.reshape(-1)
            post_flat = post_indices.reshape(-1)
            for k, (l_value, u_value) in enumerate(zip(pre_l, pre_u)):
                z_index = pre_flat[k]
                y_index = post_flat[k]

                if u_value <= 0.0:
                    # y is already fixed to zero by its variable bound.
                    continue

                if l_value >= 0.0:
                    add_eq([y_index, z_index], [1.0, -1.0], 0.0)
                    continue

                # y >= z  <=>  z - y <= 0
                add_ub([z_index, y_index], [1.0, -1.0], 0.0)

                # y <= lambda * (z - l)
                slope = u_value / max(u_value - l_value, 1e-15)
                add_ub(
                    [y_index, z_index],
                    [1.0, -slope],
                    -slope * l_value,
                )

            current_indices = post_indices
            continue

        raise RuntimeError('Unexpected LP record kind: {}'.format(kind))

    logit_indices = current_indices.reshape(-1)
    if logit_indices.size != output_size:
        raise RuntimeError('Unexpected output size while building LP.')

    A_eq = sp.coo_matrix(
        (eq_data, (eq_row, eq_col)),
        shape=(eq_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    b_eq = np.asarray(eq_rhs, dtype=np.float64)
    A_ub_base = sp.coo_matrix(
        (ub_data, (ub_row, ub_col)),
        shape=(ub_count, variable_count),
        dtype=np.float64,
    ).tocsr()
    b_ub_base = np.asarray(ub_rhs, dtype=np.float64)

    # ------------------------------------------------------------------
    # Geometry helpers for sound 2-ReLU cuts.
    # ------------------------------------------------------------------
    first_A_np = first_relu_A.detach().cpu().numpy()
    first_b_np = first_relu_b.detach().cpu().numpy()
    input_l_np = input_l.detach().cpu().numpy()
    input_u_np = input_u.detach().cpu().numpy()
    first_l_np = first_relu_l.detach().cpu().numpy()
    first_u_np = first_relu_u.detach().cpu().numpy()
    unstable_mask = (first_l_np < 0.0) & (first_u_np > 0.0)

    angles = np.linspace(
        0.0, 2.0 * np.pi, SUPPORT_DIRECTIONS, endpoint=False
    )
    support_directions = np.stack(
        [np.cos(angles), np.sin(angles)], axis=1
    )

    pair_cut_cache = {}

    def polygon_vertices(normals, rhs):
        vertices = []
        count = normals.shape[0]
        for i in range(count):
            for j in range(i + 1, count):
                matrix = np.stack([normals[i], normals[j]], axis=0)
                determinant = np.linalg.det(matrix)
                if abs(determinant) <= GEOM_TOL:
                    continue
                candidate = np.linalg.solve(matrix, np.array([rhs[i], rhs[j]]))
                if np.all(normals @ candidate <= rhs + 2e-7):
                    vertices.append(candidate)

        if not vertices:
            return np.empty((0, 2), dtype=np.float64)

        vertices = np.asarray(vertices, dtype=np.float64)
        rounded = np.round(vertices, decimals=10)
        _, unique_index = np.unique(rounded, axis=0, return_index=True)
        vertices = vertices[np.sort(unique_index)]

        if vertices.shape[0] > 2:
            center = vertices.mean(axis=0)
            order = np.argsort(
                np.arctan2(vertices[:, 1] - center[1],
                           vertices[:, 0] - center[0])
            )
            vertices = vertices[order]
        return vertices

    def build_pair_cuts(first_i, first_j):
        key = (min(first_i, first_j), max(first_i, first_j))
        if key in pair_cut_cache:
            return pair_cut_cache[key]

        ai = first_A_np[first_i]
        aj = first_A_np[first_j]
        bi = first_b_np[first_i]
        bj = first_b_np[first_j]

        normals = support_directions.copy()
        rhs = []
        for direction in normals:
            combined = direction[0] * ai + direction[1] * aj
            support = (
                np.maximum(combined, 0.0) @ input_u_np
                + np.minimum(combined, 0.0) @ input_l_np
                + direction[0] * bi
                + direction[1] * bj
            )
            rhs.append(
                support
                + SUPPORT_SLACK * (1.0 + abs(float(support)))
            )
        rhs = np.asarray(rhs, dtype=np.float64)

        lifted_points = []
        for sign_i in (0, 1):
            for sign_j in (0, 1):
                extra_normals = []
                extra_rhs = []

                # sign=0 means z <= 0; sign=1 means z >= 0.
                extra_normals.append([1.0, 0.0] if sign_i == 0 else [-1.0, 0.0])
                extra_rhs.append(0.0)
                extra_normals.append([0.0, 1.0] if sign_j == 0 else [0.0, -1.0])
                extra_rhs.append(0.0)

                region_normals = np.vstack(
                    [normals, np.asarray(extra_normals, dtype=np.float64)]
                )
                region_rhs = np.concatenate(
                    [rhs, np.asarray(extra_rhs, dtype=np.float64)]
                )
                region_vertices = polygon_vertices(region_normals, region_rhs)

                for vertex in region_vertices:
                    lifted_points.append([
                        vertex[0],
                        vertex[1],
                        max(0.0, vertex[0]),
                        max(0.0, vertex[1]),
                    ])

        if len(lifted_points) < 5:
            pair_cut_cache[key] = []
            return []

        points = np.asarray(lifted_points, dtype=np.float64)
        rounded = np.round(points, decimals=10)
        _, unique_index = np.unique(rounded, axis=0, return_index=True)
        points = points[np.sort(unique_index)]

        # A full four-dimensional hull is needed for useful facet equations.
        centered = points - points.mean(axis=0, keepdims=True)
        if np.linalg.matrix_rank(centered, tol=1e-9) < 4:
            pair_cut_cache[key] = []
            return []

        try:
            hull = ConvexHull(points, qhull_options='QJ')
        except QhullError:
            pair_cut_cache[key] = []
            return []

        cuts = []
        seen = set()
        for equation in hull.equations:
            normal = equation[:-1].astype(np.float64)
            offset = float(equation[-1])
            norm = np.linalg.norm(normal)
            if norm <= 1e-12:
                continue
            normal = normal / norm
            offset = offset / norm

            # Qhull represents inside as normal @ p + offset <= 0.
            violation = float(np.max(points @ normal + offset))
            rhs_value = -offset + max(0.0, violation)
            rhs_value += CUT_SLACK * (
                1.0 + abs(rhs_value) + np.linalg.norm(normal, ord=1)
            )

            signature = tuple(np.round(
                np.concatenate([normal, [rhs_value]]), decimals=8
            ))
            if signature in seen:
                continue
            seen.add(signature)
            cuts.append((normal, rhs_value))

        pair_cut_cache[key] = cuts
        return cuts

    # ------------------------------------------------------------------
    # Target-specific pair selection. Selection may be heuristic; soundness
    # depends only on the validity of the cuts that are actually added.
    # ------------------------------------------------------------------
    gradient_cache = {}

    def first_relu_gradient(target_label):
        if target_label in gradient_cache:
            return gradient_cache[target_label]

        with torch.enable_grad():
            current = x.detach().clone()
            first_activation = None
            for layer in layers:
                current = layer(current)
                if first_activation is None and isinstance(layer, torch.nn.ReLU):
                    first_activation = current
            if first_activation is None:
                gradient = torch.zeros_like(first_relu_l)
            else:
                output = current.reshape(-1)
                margin = output[true_label] - output[target_label]
                gradient = torch.autograd.grad(
                    margin,
                    first_activation,
                    retain_graph=False,
                    create_graph=False,
                )[0].reshape(-1).detach().to(dtype=dtype)

        result = gradient.cpu().numpy()
        gradient_cache[target_label] = result
        return result

    def select_pairs(target_label):
        gradient = first_relu_gradient(target_label)
        widths = np.maximum(first_u_np - first_l_np, 0.0)
        scores = np.abs(gradient) * widths
        scores[~unstable_mask] = -np.inf

        valid = np.flatnonzero(np.isfinite(scores) & (scores > 0.0))
        if valid.size < 2:
            valid = np.flatnonzero(unstable_mask)
        if valid.size < 2:
            return []

        order = valid[np.argsort(scores[valid])[::-1]]
        candidates = order[:MAX_CANDIDATES].tolist()
        remaining = candidates[:]
        pairs = []

        while len(remaining) >= 2 and len(pairs) < MAX_PAIRS:
            first_i = remaining.pop(0)
            ai = first_A_np[first_i]
            ai_norm = np.linalg.norm(ai) + 1e-12

            best_position = None
            best_value = -float('inf')
            for position, first_j in enumerate(remaining):
                aj = first_A_np[first_j]
                correlation = abs(
                    float(ai @ aj)
                    / (ai_norm * (np.linalg.norm(aj) + 1e-12))
                )
                value = (
                    (max(scores[first_i], 0.0) + 1e-12)
                    * (max(scores[first_j], 0.0) + 1e-12)
                    * (0.25 + correlation)
                )
                if value > best_value:
                    best_value = value
                    best_position = position

            if best_position is None:
                break
            first_j = remaining.pop(best_position)
            pairs.append((int(first_i), int(first_j)))

        return pairs

    # Hardest IBP margins first, so an unprovable case exits early.
    target_labels = [
        label for label in range(output_size) if label != true_label
    ]
    target_labels.sort(
        key=lambda label: float(output_low[true_label] - output_high[label])
    )

    for target_label in target_labels:
        if time.perf_counter() >= deadline:
            return False

        # A positive IBP margin is already a proof.
        ibp_margin = float(
            output_low[true_label] - output_high[target_label]
        )
        if ibp_margin > CERT_TOL:
            continue

        pair_rows = []
        pair_cols = []
        pair_data = []
        pair_rhs = []
        pair_row_count = 0

        for first_i, first_j in select_pairs(target_label):
            if time.perf_counter() >= deadline:
                return False

            cuts = build_pair_cuts(first_i, first_j)
            z_i = int(first_relu_pre_indices[first_i])
            z_j = int(first_relu_pre_indices[first_j])
            y_i = int(first_relu_post_indices[first_i])
            y_j = int(first_relu_post_indices[first_j])
            variables = [z_i, z_j, y_i, y_j]

            for normal, rhs_value in cuts:
                for variable, coefficient in zip(variables, normal):
                    if abs(coefficient) > 0.0:
                        pair_rows.append(pair_row_count)
                        pair_cols.append(variable)
                        pair_data.append(float(coefficient))
                pair_rhs.append(float(rhs_value))
                pair_row_count += 1

        if pair_row_count:
            pair_matrix = sp.coo_matrix(
                (pair_data, (pair_rows, pair_cols)),
                shape=(pair_row_count, variable_count),
                dtype=np.float64,
            ).tocsr()
            A_ub = sp.vstack([A_ub_base, pair_matrix], format='csr')
            b_ub = np.concatenate(
                [b_ub_base, np.asarray(pair_rhs, dtype=np.float64)]
            )
        else:
            A_ub = A_ub_base
            b_ub = b_ub_base

        objective = np.zeros(variable_count, dtype=np.float64)
        objective[logit_indices[true_label]] = 1.0
        objective[logit_indices[target_label]] = -1.0

        remaining = max(1.0, deadline - time.perf_counter())
        result = linprog(
            objective,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=variable_bounds,
            method='highs',
            options={'time_limit': remaining},
        )

        # Only a confirmed optimal LP result with a positive lower bound
        # constitutes a certificate. Infeasible/timeout/numerical states fail.
        if not result.success or result.fun is None:
            return False
        if float(result.fun) <= CERT_TOL:
            return False

    return True



def analyze(model, x, eps, true_label):
    """
    Two-stage verifier:
      1. Existing optimized Hybrid IBP + DeepPoly pass.
      2. If that fails, a sparse whole-network LP with selective 2-ReLU cuts.

    The old CROWN fallback and interval-reset experiments are intentionally
    removed. They did not certify additional examples in the diagnostics.
    """
    start_time = time.perf_counter()

    if _analyze_forward_hybrid(model, x, eps, true_label):
        return True

    # Keep a safety margin below the 3-minute project limit.
    remaining_budget = 168.0 - (time.perf_counter() - start_time)
    if remaining_budget < 8.0:
        return False

    return _analyze_selective_2relu_lp(
        model,
        x,
        eps,
        true_label,
        time_budget_seconds=remaining_budget,
    )


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

    result = analyze(net, inputs, eps, true_label)
    return result


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
            try:
                result = run_single_case(net_name, spec_path)
            except Exception as exc:
                print(f'{net_name}\t{os.path.basename(spec_path)}\tERROR\t{exc}')
                continue

            status = 'verified' if result else 'not verified'
            print(f'{net_name}\t{os.path.basename(spec_path)}\t{status}')

# run single case: python verifier_selective_2relu.py --net fc1 --spec ../test_cases/fc1/img0_0.09500.txt
# run all cases: python verifier_selective_2relu.py --batch --tests-dir ../test_cases

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