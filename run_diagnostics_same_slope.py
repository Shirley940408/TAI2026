"""
Usage (run from the same directory as verifier_debug.py):

    python run_diagnostics_same_slope.py --tests-dir ../test_cases --gt ../test_cases/gt.txt

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

from pure_deeppoly_same_slope import parse_spec, load_network, analyze, INPUT_SIZE, DEVICE
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

# ==================== SUMMARY ====================
# Overall: 25/50 = 50.0%
# Total wall time: 8.0s, avg 0.16s/case

# net      correct  total 
# conv1           3      5
# conv2           2      5
# conv3           1      5
# fc1             4      5
# fc2             3      5
# fc3             2      5
# fc4             3      5
# fc5             2      5
# fc6             3      5
# fc7             2      5

# No soundness violations (good -- verifier only ever loses points by being imprecise, never by being unsound).

# False negatives (gt=verified, we said not verified) -- 25 cases:
#   fc1/img2_0.11500.txt  (0.00s)
#   fc2/img0_0.08500.txt  (0.00s)
#   fc2/img1_0.08000.txt  (0.00s)
#   fc3/img0_0.04000.txt  (0.00s)
#   fc3/img1_0.04500.txt  (0.00s)
#   fc3/img2_0.04500.txt  (0.00s)
#   fc4/img0_0.17000.txt  (0.01s)
#   fc4/img3_0.03500.txt  (0.00s)
#   fc5/img1_0.12000.txt  (0.01s)
#   fc5/img3_0.10000.txt  (0.01s)
#   fc5/img4_0.14500.txt  (0.01s)
#   fc6/img3_0.08500.txt  (0.01s)
#   fc6/img4_0.05000.txt  (0.01s)
#   fc7/img1_0.14500.txt  (0.01s)
#   fc7/img2_0.17000.txt  (0.01s)
#   fc7/img4_0.15500.txt  (0.01s)
#   conv1/img0_0.13500.txt  (0.16s)
#   conv1/img1_0.13000.txt  (0.11s)
#   conv2/img0_0.13000.txt  (0.66s)
#   conv2/img2_0.16500.txt  (0.66s)
#   conv2/img4_0.16000.txt  (0.66s)
#   conv3/img1_0.18500.txt  (0.76s)
#   conv3/img2_0.23500.txt  (0.78s)
#   conv3/img3_0.24000.txt  (0.77s)
#   conv3/img4_0.24000.txt  (0.76s)