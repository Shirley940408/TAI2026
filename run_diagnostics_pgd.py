"""
Usage (run from the same directory as verifier_debug.py):

    python run_diagnostics_pgd.py --tests-dir ../test_cases --gt ../test_cases/gt.txt

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

from verifier_pgd import parse_spec, load_network, analyze, INPUT_SIZE, DEVICE
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

# (TAIvenv310) (base) shuoyan@u172-016-131-162 code % python run_diagnostics_pgd.py --tests-dir ../test_cases --gt ../test_cases/gt.txt
# [dbg] fc1/img0_0.09500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img0_0.09500.txt [forward] 0/1-heuristic min_margin=0.3885 already_certified=9/9
# [dbg] fc1/img0_0.09500.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc1/img0_0.09500.txt total_time=0.04s result=VERIFIED (forward)
# fc1/img0_0.09500.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc1/img1_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img1_0.02000.txt [forward] 0/1-heuristic min_margin=3.9139 already_certified=9/9
# [dbg] fc1/img1_0.02000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc1/img1_0.02000.txt total_time=0.05s result=VERIFIED (forward)
# fc1/img1_0.02000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc1/img2_0.11500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img2_0.11500.txt [forward] 0/1-heuristic min_margin=-5.4118 already_certified=2/9
# [dbg] fc1/img2_0.11500.txt [forward] init comparison: chosen=heuristic score=-5.8140
# [dbg] fc1/img2_0.11500.txt [forward] shared opt: min_margin -5.8140 -> -5.4153 over 18 steps
# [dbg] fc1/img2_0.11500.txt [forward] after shared phase: certified=2/9
# [dbg] fc1/img2_0.11500.txt [forward] 7 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 7, 8]
# [dbg] fc1/img2_0.11500.txt [forward] target class_idx=0: margin -0.0169 -> 0.0010 certified=True
# [dbg] fc1/img2_0.11500.txt [forward] target class_idx=1: margin -3.8579 -> -3.8486 certified=False
# [dbg] fc1/img2_0.11500.txt [forward] class_idx=1 could NOT be certified -> forward hybrid fails (0.73s)
# [dbg] fc1/img2_0.11500.txt total_time=0.76s result=NOT VERIFIED (no conv, no fallback)
# fc1/img2_0.11500.txt    gt=verified     pred=not verified       MISMATCH        0.76s
# [dbg] fc1/img3_0.09000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img3_0.09000.txt [forward] 0/1-heuristic min_margin=0.7620 already_certified=9/9
# [dbg] fc1/img3_0.09000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc1/img3_0.09000.txt total_time=0.03s result=VERIFIED (forward)
# fc1/img3_0.09000.txt    gt=verified     pred=verified   OK      0.03s
# [dbg] fc1/img4_0.14000.txt [pgd] restart=0 step=6: found adversarial example (pred=4)
# [dbg] fc1/img4_0.14000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc1/img4_0.14000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc2/img0_0.08500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img0_0.08500.txt [forward] 0/1-heuristic min_margin=-12.6937 already_certified=0/9
# [dbg] fc2/img0_0.08500.txt [forward] init comparison: chosen=heuristic score=-14.1427
# [dbg] fc2/img0_0.08500.txt [forward] shared opt: min_margin -14.1427 -> -10.6327 over 18 steps
# [dbg] fc2/img0_0.08500.txt [forward] after shared phase: certified=0/9
# [dbg] fc2/img0_0.08500.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] fc2/img0_0.08500.txt [forward] target class_idx=0: margin -6.2845 -> -5.8840 certified=False
# [dbg] fc2/img0_0.08500.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (0.14s)
# [dbg] fc2/img0_0.08500.txt total_time=0.18s result=NOT VERIFIED (no conv, no fallback)
# fc2/img0_0.08500.txt    gt=verified     pred=not verified       MISMATCH        0.18s
# [dbg] fc2/img1_0.08000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img1_0.08000.txt [forward] 0/1-heuristic min_margin=-8.1408 already_certified=1/9
# [dbg] fc2/img1_0.08000.txt [forward] init comparison: chosen=pgd-informed score=-9.3822
# [dbg] fc2/img1_0.08000.txt [forward] shared opt: min_margin -9.3822 -> -7.8201 over 18 steps
# [dbg] fc2/img1_0.08000.txt [forward] after shared phase: certified=1/9
# [dbg] fc2/img1_0.08000.txt [forward] 8 classes unresolved -> per-target opt: [1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] fc2/img1_0.08000.txt [forward] target class_idx=1: margin -0.9269 -> -0.7415 certified=False
# [dbg] fc2/img1_0.08000.txt [forward] class_idx=1 could NOT be certified -> forward hybrid fails (0.15s)
# [dbg] fc2/img1_0.08000.txt total_time=0.19s result=NOT VERIFIED (no conv, no fallback)
# fc2/img1_0.08000.txt    gt=verified     pred=not verified       MISMATCH        0.19s
# [dbg] fc2/img2_0.17000.txt [pgd] restart=0 step=9: found adversarial example (pred=8)
# [dbg] fc2/img2_0.17000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc2/img2_0.17000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc2/img3_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img3_0.01000.txt [forward] 0/1-heuristic min_margin=7.5907 already_certified=9/9
# [dbg] fc2/img3_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img3_0.01000.txt total_time=0.04s result=VERIFIED (forward)
# fc2/img3_0.01000.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc2/img4_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img4_0.01000.txt [forward] 0/1-heuristic min_margin=13.5898 already_certified=9/9
# [dbg] fc2/img4_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img4_0.01000.txt total_time=0.04s result=VERIFIED (forward)
# fc2/img4_0.01000.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc3/img0_0.04000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img0_0.04000.txt [forward] 0/1-heuristic min_margin=-1.7306 already_certified=4/9
# [dbg] fc3/img0_0.04000.txt [forward] init comparison: chosen=pgd-informed score=-1.8281
# [dbg] fc3/img0_0.04000.txt [forward] shared opt: min_margin -1.8281 -> -1.5498 over 18 steps
# [dbg] fc3/img0_0.04000.txt [forward] after shared phase: certified=5/9
# [dbg] fc3/img0_0.04000.txt [forward] 4 classes unresolved -> per-target opt: [3, 6, 7, 8]
# [dbg] fc3/img0_0.04000.txt [forward] target class_idx=3: margin -0.1162 -> -0.0156 certified=False
# [dbg] fc3/img0_0.04000.txt [forward] class_idx=3 could NOT be certified -> forward hybrid fails (0.15s)
# [dbg] fc3/img0_0.04000.txt total_time=0.19s result=NOT VERIFIED (no conv, no fallback)
# fc3/img0_0.04000.txt    gt=verified     pred=not verified       MISMATCH        0.19s
# [dbg] fc3/img1_0.04500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img1_0.04500.txt [forward] 0/1-heuristic min_margin=0.3374 already_certified=9/9
# [dbg] fc3/img1_0.04500.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc3/img1_0.04500.txt total_time=0.04s result=VERIFIED (forward)
# fc3/img1_0.04500.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc3/img2_0.04500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img2_0.04500.txt [forward] 0/1-heuristic min_margin=-1.7788 already_certified=7/9
# [dbg] fc3/img2_0.04500.txt [forward] init comparison: chosen=heuristic score=-1.9291
# [dbg] fc3/img2_0.04500.txt [forward] shared opt: min_margin -1.9291 -> -1.6698 over 18 steps
# [dbg] fc3/img2_0.04500.txt [forward] after shared phase: certified=7/9
# [dbg] fc3/img2_0.04500.txt [forward] 2 classes unresolved -> per-target opt: [4, 8]
# [dbg] fc3/img2_0.04500.txt [forward] target class_idx=4: margin -1.6675 -> -1.6250 certified=False
# [dbg] fc3/img2_0.04500.txt [forward] class_idx=4 could NOT be certified -> forward hybrid fails (0.15s)
# [dbg] fc3/img2_0.04500.txt total_time=0.19s result=NOT VERIFIED (no conv, no fallback)
# fc3/img2_0.04500.txt    gt=verified     pred=not verified       MISMATCH        0.19s
# [dbg] fc3/img3_0.05000.txt [pgd] restart=0 step=4: found adversarial example (pred=2)
# [dbg] fc3/img3_0.05000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc3/img3_0.05000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc3/img4_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img4_0.02000.txt [forward] 0/1-heuristic min_margin=6.9455 already_certified=9/9
# [dbg] fc3/img4_0.02000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc3/img4_0.02000.txt total_time=0.04s result=VERIFIED (forward)
# fc3/img4_0.02000.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc4/img0_0.17000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img0_0.17000.txt [forward] 0/1-heuristic min_margin=0.0581 already_certified=9/9
# [dbg] fc4/img0_0.17000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc4/img0_0.17000.txt total_time=0.05s result=VERIFIED (forward)
# fc4/img0_0.17000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc4/img1_0.22000.txt [pgd] restart=0 step=6: found adversarial example (pred=4)
# [dbg] fc4/img1_0.22000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc4/img1_0.22000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc4/img2_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img2_0.13000.txt [forward] 0/1-heuristic min_margin=0.4752 already_certified=9/9
# [dbg] fc4/img2_0.13000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc4/img2_0.13000.txt total_time=0.05s result=VERIFIED (forward)
# fc4/img2_0.13000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc4/img3_0.03500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img3_0.03500.txt [forward] 0/1-heuristic min_margin=-0.0066 already_certified=8/9
# [dbg] fc4/img3_0.03500.txt [forward] init comparison: chosen=heuristic score=-0.0066
# [dbg] fc4/img3_0.03500.txt [forward] shared opt: min_margin -0.0066 -> -0.0066 over 18 steps
# [dbg] fc4/img3_0.03500.txt [forward] after shared phase: certified=9/9
# [dbg] fc4/img3_0.03500.txt [forward] VERIFIED after shared phase (0.13s)
# [dbg] fc4/img3_0.03500.txt total_time=0.17s result=VERIFIED (forward)
# fc4/img3_0.03500.txt    gt=verified     pred=verified   OK      0.17s
# [dbg] fc4/img4_0.16000.txt [pgd] restart=1 step=10: found adversarial example (pred=7)
# [dbg] fc4/img4_0.16000.txt total_time=0.01s result=NOT VERIFIED (PGD found a counterexample)
# fc4/img4_0.16000.txt    gt=not verified pred=not verified       OK      0.01s
# [dbg] fc5/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img0_0.01000.txt [forward] 0/1-heuristic min_margin=5.3763 already_certified=9/9
# [dbg] fc5/img0_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc5/img0_0.01000.txt total_time=0.05s result=VERIFIED (forward)
# fc5/img0_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc5/img1_0.12000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img1_0.12000.txt [forward] 0/1-heuristic min_margin=-8.8699 already_certified=0/9
# [dbg] fc5/img1_0.12000.txt [forward] init comparison: chosen=heuristic score=-9.5113
# [dbg] fc5/img1_0.12000.txt [forward] shared opt: min_margin -9.5113 -> -7.4857 over 18 steps
# [dbg] fc5/img1_0.12000.txt [forward] after shared phase: certified=0/9
# [dbg] fc5/img1_0.12000.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] fc5/img1_0.12000.txt [forward] target class_idx=0: margin -3.3028 -> -3.0246 certified=False
# [dbg] fc5/img1_0.12000.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (0.22s)
# [dbg] fc5/img1_0.12000.txt total_time=0.26s result=NOT VERIFIED (no conv, no fallback)
# fc5/img1_0.12000.txt    gt=verified     pred=not verified       MISMATCH        0.26s
# [dbg] fc5/img2_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img2_0.01000.txt [forward] 0/1-heuristic min_margin=7.0883 already_certified=9/9
# [dbg] fc5/img2_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc5/img2_0.01000.txt total_time=0.05s result=VERIFIED (forward)
# fc5/img2_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc5/img3_0.10000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img3_0.10000.txt [forward] 0/1-heuristic min_margin=-5.6614 already_certified=1/9
# [dbg] fc5/img3_0.10000.txt [forward] init comparison: chosen=pgd-informed score=-6.3354
# [dbg] fc5/img3_0.10000.txt [forward] shared opt: min_margin -6.3354 -> -4.9035 over 18 steps
# [dbg] fc5/img3_0.10000.txt [forward] after shared phase: certified=2/9
# [dbg] fc5/img3_0.10000.txt [forward] 7 classes unresolved -> per-target opt: [1, 2, 3, 4, 5, 7, 8]
# [dbg] fc5/img3_0.10000.txt [forward] target class_idx=1: margin -4.8959 -> -4.8494 certified=False
# [dbg] fc5/img3_0.10000.txt [forward] class_idx=1 could NOT be certified -> forward hybrid fails (0.22s)
# [dbg] fc5/img3_0.10000.txt total_time=0.26s result=NOT VERIFIED (no conv, no fallback)
# fc5/img3_0.10000.txt    gt=verified     pred=not verified       MISMATCH        0.26s
# [dbg] fc5/img4_0.14500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img4_0.14500.txt [forward] 0/1-heuristic min_margin=-7.6507 already_certified=0/9
# [dbg] fc5/img4_0.14500.txt [forward] init comparison: chosen=heuristic score=-8.9061
# [dbg] fc5/img4_0.14500.txt [forward] shared opt: min_margin -8.9061 -> -6.3897 over 18 steps
# [dbg] fc5/img4_0.14500.txt [forward] after shared phase: certified=0/9
# [dbg] fc5/img4_0.14500.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] fc5/img4_0.14500.txt [forward] target class_idx=0: margin -2.7194 -> -2.2241 certified=False
# [dbg] fc5/img4_0.14500.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (0.22s)
# [dbg] fc5/img4_0.14500.txt total_time=0.26s result=NOT VERIFIED (no conv, no fallback)
# fc5/img4_0.14500.txt    gt=verified     pred=not verified       MISMATCH        0.26s
# [dbg] fc6/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img0_0.01000.txt [forward] 0/1-heuristic min_margin=17.9055 already_certified=9/9
# [dbg] fc6/img0_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc6/img0_0.01000.txt total_time=0.05s result=VERIFIED (forward)
# fc6/img0_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc6/img1_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img1_0.01000.txt [forward] 0/1-heuristic min_margin=11.0869 already_certified=9/9
# [dbg] fc6/img1_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc6/img1_0.01000.txt total_time=0.05s result=VERIFIED (forward)
# fc6/img1_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc6/img2_0.07000.txt [pgd] restart=0 step=6: found adversarial example (pred=6)
# [dbg] fc6/img2_0.07000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc6/img2_0.07000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc6/img3_0.08500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img3_0.08500.txt [forward] 0/1-heuristic min_margin=-9.1153 already_certified=1/9
# [dbg] fc6/img3_0.08500.txt [forward] init comparison: chosen=heuristic score=-9.9609
# [dbg] fc6/img3_0.08500.txt [forward] shared opt: min_margin -9.9609 -> -6.1386 over 18 steps
# [dbg] fc6/img3_0.08500.txt [forward] after shared phase: certified=3/9
# [dbg] fc6/img3_0.08500.txt [forward] 6 classes unresolved -> per-target opt: [0, 1, 2, 3, 6, 8]
# [dbg] fc6/img3_0.08500.txt [forward] target class_idx=0: margin -0.0013 -> 0.1141 certified=True
# [dbg] fc6/img3_0.08500.txt [forward] target class_idx=1: margin -6.0586 -> -5.8134 certified=False
# [dbg] fc6/img3_0.08500.txt [forward] class_idx=1 could NOT be certified -> forward hybrid fails (0.31s)
# [dbg] fc6/img3_0.08500.txt total_time=0.36s result=NOT VERIFIED (no conv, no fallback)
# fc6/img3_0.08500.txt    gt=verified     pred=not verified       MISMATCH        0.36s
# [dbg] fc6/img4_0.05000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img4_0.05000.txt [forward] 0/1-heuristic min_margin=-1.8313 already_certified=8/9
# [dbg] fc6/img4_0.05000.txt [forward] init comparison: chosen=pgd-informed score=-1.0004
# [dbg] fc6/img4_0.05000.txt [forward] shared opt: min_margin -1.0004 -> -0.2510 over 18 steps
# [dbg] fc6/img4_0.05000.txt [forward] after shared phase: certified=8/9
# [dbg] fc6/img4_0.05000.txt [forward] 1 classes unresolved -> per-target opt: [8]
# [dbg] fc6/img4_0.05000.txt [forward] target class_idx=8: margin -0.2461 -> -0.2346 certified=False
# [dbg] fc6/img4_0.05000.txt [forward] class_idx=8 could NOT be certified -> forward hybrid fails (0.30s)
# [dbg] fc6/img4_0.05000.txt total_time=0.34s result=NOT VERIFIED (no conv, no fallback)
# fc6/img4_0.05000.txt    gt=verified     pred=not verified       MISMATCH        0.34s
# [dbg] fc7/img0_0.50000.txt [pgd] restart=0 step=2: found adversarial example (pred=9)
# [dbg] fc7/img0_0.50000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc7/img0_0.50000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc7/img1_0.14500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img1_0.14500.txt [forward] 0/1-heuristic min_margin=1.8174 already_certified=9/9
# [dbg] fc7/img1_0.14500.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc7/img1_0.14500.txt total_time=0.06s result=VERIFIED (forward)
# fc7/img1_0.14500.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc7/img2_0.17000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img2_0.17000.txt [forward] 0/1-heuristic min_margin=0.0381 already_certified=9/9
# [dbg] fc7/img2_0.17000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc7/img2_0.17000.txt total_time=0.06s result=VERIFIED (forward)
# fc7/img2_0.17000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc7/img3_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img3_0.02000.txt [forward] 0/1-heuristic min_margin=1.5138 already_certified=9/9
# [dbg] fc7/img3_0.02000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc7/img3_0.02000.txt total_time=0.06s result=VERIFIED (forward)
# fc7/img3_0.02000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc7/img4_0.15500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img4_0.15500.txt [forward] 0/1-heuristic min_margin=0.1012 already_certified=9/9
# [dbg] fc7/img4_0.15500.txt [forward] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc7/img4_0.15500.txt total_time=0.06s result=VERIFIED (forward)
# fc7/img4_0.15500.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] conv1/img0_0.13500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img0_0.13500.txt [forward] 0/1-heuristic min_margin=-4.1766 already_certified=4/9
# [dbg] conv1/img0_0.13500.txt [forward] init comparison: chosen=heuristic score=-6.0855
# [dbg] conv1/img0_0.13500.txt [forward] shared opt: min_margin -6.0855 -> -3.1163 over 18 steps
# [dbg] conv1/img0_0.13500.txt [forward] after shared phase: certified=4/9
# [dbg] conv1/img0_0.13500.txt [forward] 5 classes unresolved -> per-target opt: [0, 1, 2, 4, 8]
# [dbg] conv1/img0_0.13500.txt [forward] target class_idx=0: margin -0.9942 -> 0.0291 certified=True
# [dbg] conv1/img0_0.13500.txt [forward] target class_idx=1: margin -3.1019 -> -2.9820 certified=False
# [dbg] conv1/img0_0.13500.txt [forward] class_idx=1 could NOT be certified -> forward hybrid fails (6.10s)
# [dbg] conv1/img0_0.13500.txt [crown] starting, time_budget_seconds=161.8
# [dbg] conv1/img0_0.13500.txt [crown] target_label=1: best_margin=-3.6797 certified=False
# [dbg] conv1/img0_0.13500.txt [crown] target_label=1 could NOT be certified -> NOT VERIFIED (0.03s)
# [dbg] conv1/img0_0.13500.txt total_time=6.20s result=NOT VERIFIED (crown fallback)
# conv1/img0_0.13500.txt  gt=verified     pred=not verified       MISMATCH        6.20s
# [dbg] conv1/img1_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img1_0.13000.txt [forward] 0/1-heuristic min_margin=-1.5769 already_certified=7/9
# [dbg] conv1/img1_0.13000.txt [forward] init comparison: chosen=heuristic score=-2.9094
# [dbg] conv1/img1_0.13000.txt [forward] VERIFIED during shared opt at step 6 (1.39s)
# [dbg] conv1/img1_0.13000.txt total_time=1.46s result=VERIFIED (forward)
# conv1/img1_0.13000.txt  gt=verified     pred=verified   OK      1.46s
# [dbg] conv1/img2_0.31000.txt [pgd] restart=0 step=3: found adversarial example (pred=9)
# [dbg] conv1/img2_0.31000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# conv1/img2_0.31000.txt  gt=not verified pred=not verified       OK      0.00s
# [dbg] conv1/img3_0.04000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img3_0.04000.txt [forward] 0/1-heuristic min_margin=8.4198 already_certified=9/9
# [dbg] conv1/img3_0.04000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.14s)
# [dbg] conv1/img3_0.04000.txt total_time=0.20s result=VERIFIED (forward)
# conv1/img3_0.04000.txt  gt=verified     pred=verified   OK      0.20s
# [dbg] conv1/img4_0.03000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img4_0.03000.txt [forward] 0/1-heuristic min_margin=8.4635 already_certified=9/9
# [dbg] conv1/img4_0.03000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.14s)
# [dbg] conv1/img4_0.03000.txt total_time=0.21s result=VERIFIED (forward)
# conv1/img4_0.03000.txt  gt=verified     pred=verified   OK      0.21s
# [dbg] conv2/img0_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img0_0.13000.txt [forward] 0/1-heuristic min_margin=-12.4037 already_certified=1/9
# [dbg] conv2/img0_0.13000.txt [forward] init comparison: chosen=heuristic score=-13.4075
# [dbg] conv2/img0_0.13000.txt [forward] shared opt: min_margin -13.4075 -> -8.8081 over 18 steps
# [dbg] conv2/img0_0.13000.txt [forward] after shared phase: certified=2/9
# [dbg] conv2/img0_0.13000.txt [forward] 7 classes unresolved -> per-target opt: [0, 2, 3, 4, 5, 6, 8]
# [dbg] conv2/img0_0.13000.txt [forward] target class_idx=0: margin -5.9907 -> -4.8078 certified=False
# [dbg] conv2/img0_0.13000.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (24.52s)
# [dbg] conv2/img0_0.13000.txt [crown] starting, time_budget_seconds=143.4
# [dbg] conv2/img0_0.13000.txt [crown] target_label=2: best_margin=-29.6345 certified=False
# [dbg] conv2/img0_0.13000.txt [crown] target_label=2 could NOT be certified -> NOT VERIFIED (0.04s)
# [dbg] conv2/img0_0.13000.txt total_time=24.65s result=NOT VERIFIED (crown fallback)
# conv2/img0_0.13000.txt  gt=verified     pred=not verified       MISMATCH        24.65s
# [dbg] conv2/img1_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img1_0.01000.txt [forward] 0/1-heuristic min_margin=7.4781 already_certified=9/9
# [dbg] conv2/img1_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.63s)
# [dbg] conv2/img1_0.01000.txt total_time=0.73s result=VERIFIED (forward)
# conv2/img1_0.01000.txt  gt=verified     pred=verified   OK      0.73s
# [dbg] conv2/img2_0.16500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img2_0.16500.txt [forward] 0/1-heuristic min_margin=-22.8440 already_certified=2/9
# [dbg] conv2/img2_0.16500.txt [forward] init comparison: chosen=heuristic score=-29.5951
# [dbg] conv2/img2_0.16500.txt [forward] shared opt: min_margin -29.5951 -> -14.0901 over 18 steps
# [dbg] conv2/img2_0.16500.txt [forward] after shared phase: certified=2/9
# [dbg] conv2/img2_0.16500.txt [forward] 7 classes unresolved -> per-target opt: [1, 2, 3, 4, 6, 7, 8]
# [dbg] conv2/img2_0.16500.txt [forward] target class_idx=1: margin -0.6564 -> 0.3428 certified=True
# [dbg] conv2/img2_0.16500.txt [forward] target class_idx=2: margin -12.3264 -> -10.0438 certified=False
# [dbg] conv2/img2_0.16500.txt [forward] class_idx=2 could NOT be certified -> forward hybrid fails (27.01s)
# [dbg] conv2/img2_0.16500.txt [crown] starting, time_budget_seconds=140.9
# [dbg] conv2/img2_0.16500.txt [crown] target_label=2: best_margin=-38.3837 certified=False
# [dbg] conv2/img2_0.16500.txt [crown] target_label=2 could NOT be certified -> NOT VERIFIED (0.04s)
# [dbg] conv2/img2_0.16500.txt total_time=27.14s result=NOT VERIFIED (crown fallback)
# conv2/img2_0.16500.txt  gt=verified     pred=not verified       MISMATCH        27.14s
# [dbg] conv2/img3_0.27000.txt [pgd] restart=1 step=23: found adversarial example (pred=3)
# [dbg] conv2/img3_0.27000.txt total_time=0.03s result=NOT VERIFIED (PGD found a counterexample)
# conv2/img3_0.27000.txt  gt=not verified pred=not verified       OK      0.03s
# [dbg] conv2/img4_0.16000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img4_0.16000.txt [forward] 0/1-heuristic min_margin=-11.2287 already_certified=5/9
# [dbg] conv2/img4_0.16000.txt [forward] init comparison: chosen=heuristic score=-17.9362
# [dbg] conv2/img4_0.16000.txt [forward] shared opt: min_margin -17.9362 -> -5.4572 over 18 steps
# [dbg] conv2/img4_0.16000.txt [forward] after shared phase: certified=6/9
# [dbg] conv2/img4_0.16000.txt [forward] 3 classes unresolved -> per-target opt: [0, 2, 4]
# [dbg] conv2/img4_0.16000.txt [forward] target class_idx=0: margin -2.6206 -> -1.0245 certified=False
# [dbg] conv2/img4_0.16000.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (24.72s)
# [dbg] conv2/img4_0.16000.txt [crown] starting, time_budget_seconds=143.2
# [dbg] conv2/img4_0.16000.txt [crown] target_label=2: best_margin=-30.0820 certified=False
# [dbg] conv2/img4_0.16000.txt [crown] target_label=2 could NOT be certified -> NOT VERIFIED (0.04s)
# [dbg] conv2/img4_0.16000.txt total_time=24.84s result=NOT VERIFIED (crown fallback)
# conv2/img4_0.16000.txt  gt=verified     pred=not verified       MISMATCH        24.84s
# [dbg] conv3/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img0_0.01000.txt [forward] 0/1-heuristic min_margin=8.9139 already_certified=9/9
# [dbg] conv3/img0_0.01000.txt [forward] VERIFIED from the 0/1 heuristic alone (0.78s)
# [dbg] conv3/img0_0.01000.txt total_time=0.88s result=VERIFIED (forward)
# conv3/img0_0.01000.txt  gt=verified     pred=verified   OK      0.88s
# [dbg] conv3/img1_0.18500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img1_0.18500.txt [forward] 0/1-heuristic min_margin=-32.6804 already_certified=0/9
# [dbg] conv3/img1_0.18500.txt [forward] init comparison: chosen=heuristic score=-35.1339
# [dbg] conv3/img1_0.18500.txt [forward] shared opt: min_margin -35.1339 -> -23.7199 over 18 steps
# [dbg] conv3/img1_0.18500.txt [forward] after shared phase: certified=0/9
# [dbg] conv3/img1_0.18500.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] conv3/img1_0.18500.txt [forward] target class_idx=0: margin -17.6974 -> -17.2087 certified=False
# [dbg] conv3/img1_0.18500.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (28.64s)
# [dbg] conv3/img1_0.18500.txt [crown] starting, time_budget_seconds=139.3
# [dbg] conv3/img1_0.18500.txt [crown] target_label=8: best_margin=-90.3050 certified=False
# [dbg] conv3/img1_0.18500.txt [crown] target_label=8 could NOT be certified -> NOT VERIFIED (0.04s)
# [dbg] conv3/img1_0.18500.txt total_time=28.78s result=NOT VERIFIED (crown fallback)
# conv3/img1_0.18500.txt  gt=verified     pred=not verified       MISMATCH        28.78s
# [dbg] conv3/img2_0.23500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img2_0.23500.txt [forward] 0/1-heuristic min_margin=-63.5339 already_certified=0/9
# [dbg] conv3/img2_0.23500.txt [forward] init comparison: chosen=heuristic score=-66.2132
# [dbg] conv3/img2_0.23500.txt [forward] shared opt: min_margin -66.2132 -> -46.9461 over 18 steps
# [dbg] conv3/img2_0.23500.txt [forward] after shared phase: certified=0/9
# [dbg] conv3/img2_0.23500.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] conv3/img2_0.23500.txt [forward] target class_idx=0: margin -39.4876 -> -38.7529 certified=False
# [dbg] conv3/img2_0.23500.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (28.51s)
# [dbg] conv3/img2_0.23500.txt [crown] starting, time_budget_seconds=139.4
# [dbg] conv3/img2_0.23500.txt [crown] target_label=8: best_margin=-142.2162 certified=False
# [dbg] conv3/img2_0.23500.txt [crown] target_label=8 could NOT be certified -> NOT VERIFIED (0.05s)
# [dbg] conv3/img2_0.23500.txt total_time=28.66s result=NOT VERIFIED (crown fallback)
# conv3/img2_0.23500.txt  gt=verified     pred=not verified       MISMATCH        28.66s
# [dbg] conv3/img3_0.24000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img3_0.24000.txt [forward] 0/1-heuristic min_margin=-72.2174 already_certified=0/9
# [dbg] conv3/img3_0.24000.txt [forward] init comparison: chosen=heuristic score=-74.0359
# [dbg] conv3/img3_0.24000.txt [forward] shared opt: min_margin -74.0359 -> -56.4247 over 18 steps
# [dbg] conv3/img3_0.24000.txt [forward] after shared phase: certified=0/9
# [dbg] conv3/img3_0.24000.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] conv3/img3_0.24000.txt [forward] target class_idx=0: margin -44.5222 -> -43.2885 certified=False
# [dbg] conv3/img3_0.24000.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (28.54s)
# [dbg] conv3/img3_0.24000.txt [crown] starting, time_budget_seconds=139.4
# [dbg] conv3/img3_0.24000.txt [crown] target_label=2: best_margin=-135.1954 certified=False
# [dbg] conv3/img3_0.24000.txt [crown] target_label=2 could NOT be certified -> NOT VERIFIED (0.05s)
# [dbg] conv3/img3_0.24000.txt total_time=28.69s result=NOT VERIFIED (crown fallback)
# conv3/img3_0.24000.txt  gt=verified     pred=not verified       MISMATCH        28.69s
# [dbg] conv3/img4_0.24000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img4_0.24000.txt [forward] 0/1-heuristic min_margin=-56.7109 already_certified=0/9
# [dbg] conv3/img4_0.24000.txt [forward] init comparison: chosen=heuristic score=-57.5903
# [dbg] conv3/img4_0.24000.txt [forward] shared opt: min_margin -57.5903 -> -46.7507 over 18 steps
# [dbg] conv3/img4_0.24000.txt [forward] after shared phase: certified=0/9
# [dbg] conv3/img4_0.24000.txt [forward] 9 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 5, 6, 7, 8]
# [dbg] conv3/img4_0.24000.txt [forward] target class_idx=0: margin -36.1933 -> -35.5793 certified=False
# [dbg] conv3/img4_0.24000.txt [forward] class_idx=0 could NOT be certified -> forward hybrid fails (28.62s)
# [dbg] conv3/img4_0.24000.txt [crown] starting, time_budget_seconds=139.3
# [dbg] conv3/img4_0.24000.txt [crown] target_label=2: best_margin=-96.7919 certified=False
# [dbg] conv3/img4_0.24000.txt [crown] target_label=2 could NOT be certified -> NOT VERIFIED (0.05s)
# [dbg] conv3/img4_0.24000.txt total_time=28.77s result=NOT VERIFIED (crown fallback)
# conv3/img4_0.24000.txt  gt=verified     pred=not verified       MISMATCH        28.77s

# ==================== SUMMARY ====================
# Overall: 32/50 = 64.0%
# Total wall time: 205.2s, avg 4.10s/case

# net      correct  total 
# conv1           4      5
# conv2           2      5
# conv3           1      5
# fc1             4      5
# fc2             3      5
# fc3             3      5
# fc4             5      5
# fc5             2      5
# fc6             3      5
# fc7             5      5

# No soundness violations (good -- verifier only ever loses points by being imprecise, never by being unsound).

# False negatives (gt=verified, we said not verified) -- 18 cases:
#   fc1/img2_0.11500.txt  (0.76s)
#   fc2/img0_0.08500.txt  (0.18s)
#   fc2/img1_0.08000.txt  (0.19s)
#   fc3/img0_0.04000.txt  (0.19s)
#   fc3/img2_0.04500.txt  (0.19s)
#   fc5/img1_0.12000.txt  (0.26s)
#   fc5/img3_0.10000.txt  (0.26s)
#   fc5/img4_0.14500.txt  (0.26s)
#   fc6/img3_0.08500.txt  (0.36s)
#   fc6/img4_0.05000.txt  (0.34s)
#   conv1/img0_0.13500.txt  (6.20s)
#   conv2/img0_0.13000.txt  (24.65s)
#   conv2/img2_0.16500.txt  (27.14s)
#   conv2/img4_0.16000.txt  (24.84s)
#   conv3/img1_0.18500.txt  (28.78s)
#   conv3/img2_0.23500.txt  (28.66s)
#   conv3/img3_0.24000.txt  (28.69s)
#   conv3/img4_0.24000.txt  (28.77s)