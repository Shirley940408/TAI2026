"""
Diagnostics runner for verifier_selective_2relu.py

Run from the same directory as verifier_selective_2relu.py, for example:

    python diagnostics_verifier_selective_2relu.py \
        --tests-dir ../test_cases \
        --gt ../test_cases/gt.txt

For every case listed in gt.txt, this script:

  1. Loads the network and test case.
  2. Runs verifier_selective_2relu.analyze().
  3. Compares the result with the ground truth.
  4. Reports the verification result and analyze() execution time.
  5. Prints overall and per-network statistics.
  6. Separately lists false negatives, soundness violations, errors, and
     cases close to or above the three-minute time limit.

Interpretation:

  * False negative:
        ground truth = verified
        verifier     = not verified

    This means the verifier is sound but not precise enough for that case.

  * Soundness violation:
        ground truth = not verified
        verifier     = verified

    This is a serious error and should be investigated immediately.

By default, one result line is printed after every test case. Use --quiet to
print only the final summary.
"""

import argparse
import os
import time
from collections import defaultdict

import torch

from verifier_selective_2relu import (
    DEVICE,
    INPUT_SIZE,
    NETWORK_NAMES,
    analyze,
    load_network,
    parse_spec,
)


VALID_GT_LABELS = {"verified", "not verified"}


def load_gt(gt_path):
    """Load rows formatted as: network_name,filename,verified|not verified."""
    rows = []

    with open(gt_path, "r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 3:
                raise ValueError(
                    f"{gt_path}:{line_number}: expected three comma-separated "
                    f"fields, got {len(parts)}: {line!r}"
                )

            net_name, filename, gt_label = parts

            if net_name not in NETWORK_NAMES:
                raise ValueError(
                    f"{gt_path}:{line_number}: unsupported network "
                    f"{net_name!r}"
                )

            if gt_label not in VALID_GT_LABELS:
                raise ValueError(
                    f"{gt_path}:{line_number}: ground-truth label must be "
                    f"'verified' or 'not verified', got {gt_label!r}"
                )

            rows.append((net_name, filename, gt_label))

    if not rows:
        raise ValueError(f"No test cases were found in {gt_path}")

    return rows


def percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return 100.0 * numerator / denominator


def format_seconds(seconds):
    return f"{seconds:.2f}s"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate verifier_selective_2relu.py against the example "
            "ground-truth file."
        )
    )
    parser.add_argument(
        "--tests-dir",
        type=str,
        default="../test_cases",
        help="Directory containing fc1, ..., conv3 test-case folders.",
    )
    parser.add_argument(
        "--gt",
        type=str,
        default="../test_cases/gt.txt",
        help="Ground-truth CSV-style file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case output and print only the final summary.",
    )
    parser.add_argument(
        "--net",
        choices=NETWORK_NAMES,
        default=None,
        help="Optionally evaluate only one network.",
    )
    parser.add_argument(
        "--slow-threshold",
        type=float,
        default=150.0,
        help=(
            "List cases taking at least this many seconds as slow. "
            "Default: 150 seconds."
        ),
    )
    args = parser.parse_args()

    rows = load_gt(args.gt)
    if args.net is not None:
        rows = [row for row in rows if row[0] == args.net]
        if not rows:
            raise ValueError(
                f"No ground-truth rows were found for network {args.net!r}"
            )

    tests_dir = os.path.abspath(args.tests_dir)

    # Statistics are intentionally kept explicit so soundness failures cannot
    # be hidden inside a single accuracy number.
    per_net = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0,
            "gt_verified": 0,
            "pred_verified": 0,
            "false_negative": 0,
            "soundness_violation": 0,
            "error": 0,
            "total_time": 0.0,
            "max_time": 0.0,
        }
    )

    false_negatives = []
    soundness_violations = []
    errors = []
    slow_cases = []
    case_results = []

    total_wall_start = time.perf_counter()

    for case_number, (net_name, filename, gt_label) in enumerate(rows, start=1):
        tag = f"{net_name}/{filename}"
        spec_path = os.path.join(tests_dir, net_name, filename)
        stats = per_net[net_name]
        stats["total"] += 1
        stats["gt_verified"] += int(gt_label == "verified")

        if not os.path.isfile(spec_path):
            message = f"missing test-case file: {spec_path}"
            stats["error"] += 1
            errors.append((tag, message))

            if not args.quiet:
                print(
                    f"[{case_number:02d}/{len(rows):02d}] {tag}\t"
                    f"gt={gt_label}\tpred=ERROR\t{message}",
                    flush=True,
                )
            continue

        try:
            true_label, pixel_values, eps = parse_spec(spec_path)
            net = load_network(net_name)
            net.eval()

            inputs = torch.tensor(
                pixel_values,
                dtype=torch.float32,
                device=DEVICE,
            ).view(1, 1, INPUT_SIZE, INPUT_SIZE)

            with torch.no_grad():
                outputs = net(inputs)
                clean_prediction = int(outputs.argmax(dim=1).item())

            if clean_prediction != true_label:
                raise AssertionError(
                    f"clean prediction {clean_prediction} does not match "
                    f"spec label {true_label}"
                )

            # Match the original diagnostics script: time analyze() itself,
            # excluding model loading and the clean-prediction sanity check.
            start = time.perf_counter()
            result = bool(analyze(net, inputs, eps, true_label))
            elapsed = time.perf_counter() - start

            pred_label = "verified" if result else "not verified"
            correct = pred_label == gt_label

            stats["correct"] += int(correct)
            stats["pred_verified"] += int(result)
            stats["total_time"] += elapsed
            stats["max_time"] = max(stats["max_time"], elapsed)

            if elapsed >= args.slow_threshold:
                slow_cases.append((tag, elapsed, gt_label, pred_label))

            classification = "OK"
            if not correct:
                if gt_label == "not verified" and pred_label == "verified":
                    classification = "SOUNDNESS VIOLATION"
                    stats["soundness_violation"] += 1
                    soundness_violations.append((tag, elapsed))
                else:
                    classification = "FALSE NEGATIVE"
                    stats["false_negative"] += 1
                    false_negatives.append((tag, elapsed))

            case_results.append(
                {
                    "tag": tag,
                    "gt": gt_label,
                    "pred": pred_label,
                    "correct": correct,
                    "elapsed": elapsed,
                    "eps": eps,
                    "true_label": true_label,
                }
            )

            if not args.quiet:
                print(
                    f"[{case_number:02d}/{len(rows):02d}] {tag}\t"
                    f"gt={gt_label}\tpred={pred_label}\t"
                    f"{classification}\t{elapsed:.2f}s",
                    flush=True,
                )

        except Exception as exception:
            stats["error"] += 1
            message = f"{type(exception).__name__}: {exception}"
            errors.append((tag, message))

            if not args.quiet:
                print(
                    f"[{case_number:02d}/{len(rows):02d}] {tag}\t"
                    f"gt={gt_label}\tpred=ERROR\t{message}",
                    flush=True,
                )

    total_wall_time = time.perf_counter() - total_wall_start

    total_cases = sum(stats["total"] for stats in per_net.values())
    total_errors = sum(stats["error"] for stats in per_net.values())
    evaluated_cases = total_cases - total_errors
    total_correct = sum(stats["correct"] for stats in per_net.values())
    total_false_negatives = sum(
        stats["false_negative"] for stats in per_net.values()
    )
    total_soundness_violations = sum(
        stats["soundness_violation"] for stats in per_net.values()
    )
    total_analyze_time = sum(
        stats["total_time"] for stats in per_net.values()
    )

    print("\n==================== OVERALL SUMMARY ====================")
    print(
        f"Correct: {total_correct}/{total_cases} "
        f"= {percentage(total_correct, total_cases):.1f}%"
    )
    print(
        f"Evaluated successfully: {evaluated_cases}/{total_cases}; "
        f"errors: {total_errors}"
    )
    print(f"False negatives: {total_false_negatives}")
    print(f"Soundness violations: {total_soundness_violations}")
    print(f"Total analyze() time: {total_analyze_time:.2f}s")
    print(f"Total wall time: {total_wall_time:.2f}s")

    if evaluated_cases:
        print(
            f"Average analyze() time: "
            f"{total_analyze_time / evaluated_cases:.2f}s/case"
        )

    print("\n==================== PER-NET SUMMARY ====================")
    header = (
        f"{'net':8s} {'correct':>8s} {'total':>6s} {'acc':>7s} "
        f"{'gt-ver':>7s} {'pred-ver':>8s} {'FN':>4s} {'SV':>4s} "
        f"{'err':>4s} {'avg-sec':>9s} {'max-sec':>9s}"
    )
    print(header)
    print("-" * len(header))

    for net_name in NETWORK_NAMES:
        if net_name not in per_net:
            continue

        stats = per_net[net_name]
        successfully_evaluated = stats["total"] - stats["error"]
        average_time = (
            stats["total_time"] / successfully_evaluated
            if successfully_evaluated
            else 0.0
        )

        print(
            f"{net_name:8s} "
            f"{stats['correct']:8d} "
            f"{stats['total']:6d} "
            f"{percentage(stats['correct'], stats['total']):6.1f}% "
            f"{stats['gt_verified']:7d} "
            f"{stats['pred_verified']:8d} "
            f"{stats['false_negative']:4d} "
            f"{stats['soundness_violation']:4d} "
            f"{stats['error']:4d} "
            f"{average_time:9.2f} "
            f"{stats['max_time']:9.2f}"
        )

    print("\n==================== SOUNDNESS CHECK ====================")
    if soundness_violations:
        print(
            "*** SOUNDNESS VIOLATIONS FOUND. Investigate these before "
            "submitting:"
        )
        for tag, elapsed in soundness_violations:
            print(f"  {tag}  ({format_seconds(elapsed)})")
    else:
        print("No soundness violations detected.")

    print(
        "\n==================== FALSE NEGATIVES "
        f"({len(false_negatives)}) ===================="
    )
    if false_negatives:
        for tag, elapsed in false_negatives:
            print(f"  {tag}  ({format_seconds(elapsed)})")
    else:
        print("None.")

    print(
        "\n==================== SLOW CASES "
        f"(>= {args.slow_threshold:.1f}s) ===================="
    )
    if slow_cases:
        slow_cases.sort(key=lambda item: item[1], reverse=True)
        for tag, elapsed, gt_label, pred_label in slow_cases:
            print(
                f"  {tag}  {elapsed:.2f}s  "
                f"gt={gt_label}, pred={pred_label}"
            )
    else:
        print("None.")

    print(
        "\n==================== ERRORS "
        f"({len(errors)}) ===================="
    )
    if errors:
        for tag, message in errors:
            print(f"  {tag}: {message}")
    else:
        print("None.")


if __name__ == "__main__":
    main()