"""
Usage (run from the same directory as verifier_debug.py):

    python run_diagnostics_pure_deeppoly.py --tests-dir ../test_cases --gt ../test_cases/gt.txt

For each of the 50 example cases this:
  1. Runs analyze() with debug=True (full trace printed as it runs).
  2. Compares the result against gt.txt.
  3. Times the case.
  4. At the end, prints a per-net summary table and lists every mismatch.

A "false negative" (gt=verified, we output not verified) just means we lost
precision on that case -- expected occasionally, not a bug by itself.
A "SOUNDNESS VIOLATION" (gt=not verified, we output verified) would mean the
verifier is unsound -- this should never happen; if it does, stop and debug
that specific case before doing anything else, since it means points get
deducted (-3) rather than just withheld (0).
"""
import argparse
import os
import time

from pure_deeppoly_ablation import parse_spec, load_network, analyze, INPUT_SIZE, DEVICE
import torch


def load_gt(gt_path):
    rows = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            net_name, filename, label = line.split(',')
            rows.append((net_name, filename, label.strip()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-dir', type=str, default='../test_cases')
    parser.add_argument('--gt', type=str, default='../test_cases/gt.txt')
    parser.add_argument('--quiet', action='store_true', help='Suppress the per-case [dbg] trace, only show the summary')
    args = parser.parse_args()

    rows = load_gt(args.gt)

    per_net_total = {}
    per_net_correct = {}
    mismatches = []
    soundness_violations = []
    total_time = 0.0

    for net_name, filename, gt_label in rows:
        spec_path = os.path.join(args.tests_dir, net_name, filename)
        tag = f'{net_name}/{filename}'

        true_label, pixel_values, eps = parse_spec(spec_path)
        net = load_network(net_name)
        inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
        outs = net(inputs)
        pred_label = outs.max(dim=1)[1].item()
        assert pred_label == true_label, f'{tag}: predicted label mismatch, spec file itself is inconsistent'

        start = time.perf_counter()
        result = analyze(net, inputs, eps, true_label, debug=not args.quiet, tag=tag)
        elapsed = time.perf_counter() - start
        total_time += elapsed

        pred_label_str = 'verified' if result else 'not verified'
        correct = pred_label_str == gt_label

        per_net_total[net_name] = per_net_total.get(net_name, 0) + 1
        per_net_correct[net_name] = per_net_correct.get(net_name, 0) + (1 if correct else 0)

        print(f'{tag}\tgt={gt_label}\tpred={pred_label_str}\t{"OK" if correct else "MISMATCH"}\t{elapsed:.2f}s')

        if not correct:
            if gt_label == 'not verified' and pred_label_str == 'verified':
                soundness_violations.append(tag)
            else:
                mismatches.append((tag, elapsed))

    print('\n==================== SUMMARY ====================')
    total = sum(per_net_total.values())
    correct = sum(per_net_correct.values())
    print(f'Overall: {correct}/{total} = {100.0 * correct / total:.1f}%')
    print(f'Total wall time: {total_time:.1f}s, avg {total_time / total:.2f}s/case')
    print()
    print(f'{"net":8s} {"correct":8s} {"total":6s}')
    for net_name in sorted(per_net_total):
        print(f'{net_name:8s} {per_net_correct[net_name]:8d} {per_net_total[net_name]:6d}')

    print()
    if soundness_violations:
        print(f'*** SOUNDNESS VIOLATIONS (fix these FIRST, -3 pts each): {soundness_violations}')
    else:
        print('No soundness violations (good -- verifier only ever loses points by being imprecise, never by being unsound).')

    print()
    print(f'False negatives (gt=verified, we said not verified) -- {len(mismatches)} cases:')
    for tag, elapsed in mismatches:
        print(f'  {tag}  ({elapsed:.2f}s)')


if __name__ == '__main__':
    main()