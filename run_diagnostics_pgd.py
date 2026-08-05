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
# [dbg] fc1/img0_0.09500.txt [deeppoly] unstable per layer=[20] total=20
# [dbg] fc1/img0_0.09500.txt [deeppoly] 0/1-heuristic min_margin=0.3885 already_certified=9/9
# [dbg] fc1/img0_0.09500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc1/img0_0.09500.txt total_time=0.05s result=VERIFIED
# fc1/img0_0.09500.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc1/img1_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img1_0.02000.txt [deeppoly] unstable per layer=[11] total=11
# [dbg] fc1/img1_0.02000.txt [deeppoly] 0/1-heuristic min_margin=3.9139 already_certified=9/9
# [dbg] fc1/img1_0.02000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc1/img1_0.02000.txt total_time=0.04s result=VERIFIED
# fc1/img1_0.02000.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc1/img2_0.11500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img2_0.11500.txt [deeppoly] unstable per layer=[35] total=35
# [dbg] fc1/img2_0.11500.txt [deeppoly] 0/1-heuristic min_margin=-5.4118 already_certified=2/9
# [dbg] fc1/img2_0.11500.txt [deeppoly] init comparison: chosen=heuristic score=-5.8140
# [dbg] fc1/img2_0.11500.txt [deeppoly] shared opt: min_margin -5.8140 -> -5.4153 over 18 steps
# [dbg] fc1/img2_0.11500.txt [deeppoly] after shared phase: certified=2/9
# [dbg] fc1/img2_0.11500.txt [deeppoly] 7 classes unresolved -> per-target opt: [0, 1, 2, 3, 4, 7, 8]
# [dbg] fc1/img2_0.11500.txt [deeppoly] target class_idx=0: margin -0.0169 -> 0.0010 certified=True
# [dbg] fc1/img2_0.11500.txt [deeppoly] target class_idx=1: margin -3.8579 -> -3.8486 certified=False
# [dbg] fc1/img2_0.11500.txt [deeppoly] class_idx=1 could NOT be certified -> NOT VERIFIED (0.83s)
# [dbg] fc1/img2_0.11500.txt total_time=0.86s result=NOT VERIFIED
# fc1/img2_0.11500.txt    gt=verified     pred=not verified       MISMATCH        0.86s
# [dbg] fc1/img3_0.09000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc1/img3_0.09000.txt [deeppoly] unstable per layer=[23] total=23
# [dbg] fc1/img3_0.09000.txt [deeppoly] 0/1-heuristic min_margin=0.7620 already_certified=9/9
# [dbg] fc1/img3_0.09000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc1/img3_0.09000.txt total_time=0.04s result=VERIFIED
# fc1/img3_0.09000.txt    gt=verified     pred=verified   OK      0.04s
# [dbg] fc1/img4_0.14000.txt [pgd] restart=0 step=6: found adversarial example (pred=4)
# [dbg] fc1/img4_0.14000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc1/img4_0.14000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc2/img0_0.08500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img0_0.08500.txt [deeppoly] unstable per layer=[53, 27] total=80
# [dbg] fc2/img0_0.08500.txt [deeppoly] 0/1-heuristic min_margin=1.4766 already_certified=9/9
# [dbg] fc2/img0_0.08500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img0_0.08500.txt total_time=0.05s result=VERIFIED
# fc2/img0_0.08500.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc2/img1_0.08000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img1_0.08000.txt [deeppoly] unstable per layer=[44, 30] total=74
# [dbg] fc2/img1_0.08000.txt [deeppoly] 0/1-heuristic min_margin=0.2103 already_certified=9/9
# [dbg] fc2/img1_0.08000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img1_0.08000.txt total_time=0.05s result=VERIFIED
# fc2/img1_0.08000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc2/img2_0.17000.txt [pgd] restart=1 step=11: found adversarial example (pred=8)
# [dbg] fc2/img2_0.17000.txt total_time=0.01s result=NOT VERIFIED (PGD found a counterexample)
# fc2/img2_0.17000.txt    gt=not verified pred=not verified       OK      0.01s
# [dbg] fc2/img3_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img3_0.01000.txt [deeppoly] unstable per layer=[11, 3] total=14
# [dbg] fc2/img3_0.01000.txt [deeppoly] 0/1-heuristic min_margin=7.7144 already_certified=9/9
# [dbg] fc2/img3_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img3_0.01000.txt total_time=0.05s result=VERIFIED
# fc2/img3_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc2/img4_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc2/img4_0.01000.txt [deeppoly] unstable per layer=[5, 5] total=10
# [dbg] fc2/img4_0.01000.txt [deeppoly] 0/1-heuristic min_margin=13.7592 already_certified=9/9
# [dbg] fc2/img4_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc2/img4_0.01000.txt total_time=0.05s result=VERIFIED
# fc2/img4_0.01000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc3/img0_0.04000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img0_0.04000.txt [deeppoly] unstable per layer=[25, 26] total=51
# [dbg] fc3/img0_0.04000.txt [deeppoly] 0/1-heuristic min_margin=-1.2169 already_certified=8/9
# [dbg] fc3/img0_0.04000.txt [deeppoly] init comparison: chosen=pgd-informed score=-1.3779
# [dbg] fc3/img0_0.04000.txt [deeppoly] shared opt: min_margin -1.3779 -> -1.1252 over 18 steps
# [dbg] fc3/img0_0.04000.txt [deeppoly] after shared phase: certified=8/9
# [dbg] fc3/img0_0.04000.txt [deeppoly] 1 classes unresolved -> per-target opt: [7]
# [dbg] fc3/img0_0.04000.txt [deeppoly] target class_idx=7: margin -1.1239 -> -1.1161 certified=False
# [dbg] fc3/img0_0.04000.txt [deeppoly] class_idx=7 could NOT be certified -> NOT VERIFIED (0.25s)
# [dbg] fc3/img0_0.04000.txt total_time=0.29s result=NOT VERIFIED
# fc3/img0_0.04000.txt    gt=verified     pred=not verified       MISMATCH        0.29s
# [dbg] fc3/img1_0.04500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img1_0.04500.txt [deeppoly] unstable per layer=[13, 20] total=33
# [dbg] fc3/img1_0.04500.txt [deeppoly] 0/1-heuristic min_margin=0.7887 already_certified=9/9
# [dbg] fc3/img1_0.04500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc3/img1_0.04500.txt total_time=0.05s result=VERIFIED
# fc3/img1_0.04500.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc3/img2_0.04500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img2_0.04500.txt [deeppoly] unstable per layer=[14, 21] total=35
# [dbg] fc3/img2_0.04500.txt [deeppoly] 0/1-heuristic min_margin=-1.0127 already_certified=7/9
# [dbg] fc3/img2_0.04500.txt [deeppoly] init comparison: chosen=pgd-informed score=-1.0475
# [dbg] fc3/img2_0.04500.txt [deeppoly] shared opt: min_margin -1.0475 -> -0.9723 over 18 steps
# [dbg] fc3/img2_0.04500.txt [deeppoly] after shared phase: certified=7/9
# [dbg] fc3/img2_0.04500.txt [deeppoly] 2 classes unresolved -> per-target opt: [4, 8]
# [dbg] fc3/img2_0.04500.txt [deeppoly] target class_idx=4: margin -0.9710 -> -0.9413 certified=False
# [dbg] fc3/img2_0.04500.txt [deeppoly] class_idx=4 could NOT be certified -> NOT VERIFIED (0.24s)
# [dbg] fc3/img2_0.04500.txt total_time=0.29s result=NOT VERIFIED
# fc3/img2_0.04500.txt    gt=verified     pred=not verified       MISMATCH        0.29s
# [dbg] fc3/img3_0.05000.txt [pgd] restart=0 step=5: found adversarial example (pred=2)
# [dbg] fc3/img3_0.05000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc3/img3_0.05000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc3/img4_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc3/img4_0.02000.txt [deeppoly] unstable per layer=[3, 5] total=8
# [dbg] fc3/img4_0.02000.txt [deeppoly] 0/1-heuristic min_margin=7.0234 already_certified=9/9
# [dbg] fc3/img4_0.02000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.00s)
# [dbg] fc3/img4_0.02000.txt total_time=0.05s result=VERIFIED
# fc3/img4_0.02000.txt    gt=verified     pred=verified   OK      0.05s
# [dbg] fc4/img0_0.17000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img0_0.17000.txt [deeppoly] unstable per layer=[34, 9, 8] total=51
# [dbg] fc4/img0_0.17000.txt [deeppoly] 0/1-heuristic min_margin=0.0630 already_certified=9/9
# [dbg] fc4/img0_0.17000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc4/img0_0.17000.txt total_time=0.06s result=VERIFIED
# fc4/img0_0.17000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc4/img1_0.22000.txt [pgd] restart=0 step=4: found adversarial example (pred=4)
# [dbg] fc4/img1_0.22000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc4/img1_0.22000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc4/img2_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img2_0.13000.txt [deeppoly] unstable per layer=[6, 2, 4] total=12
# [dbg] fc4/img2_0.13000.txt [deeppoly] 0/1-heuristic min_margin=0.4752 already_certified=9/9
# [dbg] fc4/img2_0.13000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc4/img2_0.13000.txt total_time=0.06s result=VERIFIED
# fc4/img2_0.13000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc4/img3_0.03500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc4/img3_0.03500.txt [deeppoly] unstable per layer=[2, 3, 2] total=7
# [dbg] fc4/img3_0.03500.txt [deeppoly] 0/1-heuristic min_margin=-0.0066 already_certified=8/9
# [dbg] fc4/img3_0.03500.txt [deeppoly] init comparison: chosen=heuristic score=-0.0066
# [dbg] fc4/img3_0.03500.txt [deeppoly] VERIFIED during shared opt at step 17 (0.22s)
# [dbg] fc4/img3_0.03500.txt total_time=0.27s result=VERIFIED
# fc4/img3_0.03500.txt    gt=verified     pred=verified   OK      0.27s
# [dbg] fc4/img4_0.16000.txt [pgd] restart=0 step=8: found adversarial example (pred=7)
# [dbg] fc4/img4_0.16000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc4/img4_0.16000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc5/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img0_0.01000.txt [deeppoly] unstable per layer=[2, 2, 2] total=6
# [dbg] fc5/img0_0.01000.txt [deeppoly] 0/1-heuristic min_margin=5.3989 already_certified=9/9
# [dbg] fc5/img0_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc5/img0_0.01000.txt total_time=0.06s result=VERIFIED
# fc5/img0_0.01000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc5/img1_0.12000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img1_0.12000.txt [deeppoly] unstable per layer=[17, 34, 37] total=88
# [dbg] fc5/img1_0.12000.txt [deeppoly] 0/1-heuristic min_margin=-0.2372 already_certified=8/9
# [dbg] fc5/img1_0.12000.txt [deeppoly] init comparison: chosen=heuristic score=-0.5620
# [dbg] fc5/img1_0.12000.txt [deeppoly] shared opt: min_margin -0.5620 -> -0.0335 over 18 steps
# [dbg] fc5/img1_0.12000.txt [deeppoly] after shared phase: certified=8/9
# [dbg] fc5/img1_0.12000.txt [deeppoly] 1 classes unresolved -> per-target opt: [2]
# [dbg] fc5/img1_0.12000.txt [deeppoly] target class_idx=2: margin -0.0253 -> -0.0112 certified=False
# [dbg] fc5/img1_0.12000.txt [deeppoly] class_idx=2 could NOT be certified -> NOT VERIFIED (0.39s)
# [dbg] fc5/img1_0.12000.txt total_time=0.44s result=NOT VERIFIED
# fc5/img1_0.12000.txt    gt=verified     pred=not verified       MISMATCH        0.44s
# [dbg] fc5/img2_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img2_0.01000.txt [deeppoly] unstable per layer=[3, 3, 2] total=8
# [dbg] fc5/img2_0.01000.txt [deeppoly] 0/1-heuristic min_margin=7.1424 already_certified=9/9
# [dbg] fc5/img2_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc5/img2_0.01000.txt total_time=0.06s result=VERIFIED
# fc5/img2_0.01000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc5/img3_0.10000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img3_0.10000.txt [deeppoly] unstable per layer=[24, 30, 35] total=89
# [dbg] fc5/img3_0.10000.txt [deeppoly] 0/1-heuristic min_margin=0.6753 already_certified=9/9
# [dbg] fc5/img3_0.10000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc5/img3_0.10000.txt total_time=0.06s result=VERIFIED
# fc5/img3_0.10000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc5/img4_0.14500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc5/img4_0.14500.txt [deeppoly] unstable per layer=[35, 37, 41] total=113
# [dbg] fc5/img4_0.14500.txt [deeppoly] 0/1-heuristic min_margin=0.3477 already_certified=9/9
# [dbg] fc5/img4_0.14500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc5/img4_0.14500.txt total_time=0.06s result=VERIFIED
# fc5/img4_0.14500.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc6/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img0_0.01000.txt [deeppoly] unstable per layer=[4, 3, 3, 6] total=16
# [dbg] fc6/img0_0.01000.txt [deeppoly] 0/1-heuristic min_margin=18.4369 already_certified=9/9
# [dbg] fc6/img0_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc6/img0_0.01000.txt total_time=0.07s result=VERIFIED
# fc6/img0_0.01000.txt    gt=verified     pred=verified   OK      0.07s
# [dbg] fc6/img1_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img1_0.01000.txt [deeppoly] unstable per layer=[4, 5, 4, 4] total=17
# [dbg] fc6/img1_0.01000.txt [deeppoly] 0/1-heuristic min_margin=11.3757 already_certified=9/9
# [dbg] fc6/img1_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc6/img1_0.01000.txt total_time=0.06s result=VERIFIED
# fc6/img1_0.01000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc6/img2_0.07000.txt [pgd] restart=0 step=6: found adversarial example (pred=6)
# [dbg] fc6/img2_0.07000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc6/img2_0.07000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc6/img3_0.08500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img3_0.08500.txt [deeppoly] unstable per layer=[17, 19, 17, 27] total=80
# [dbg] fc6/img3_0.08500.txt [deeppoly] 0/1-heuristic min_margin=1.5795 already_certified=9/9
# [dbg] fc6/img3_0.08500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc6/img3_0.08500.txt total_time=0.06s result=VERIFIED
# fc6/img3_0.08500.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc6/img4_0.05000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc6/img4_0.05000.txt [deeppoly] unstable per layer=[11, 14, 11, 16] total=52
# [dbg] fc6/img4_0.05000.txt [deeppoly] 0/1-heuristic min_margin=3.4279 already_certified=9/9
# [dbg] fc6/img4_0.05000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc6/img4_0.05000.txt total_time=0.06s result=VERIFIED
# fc6/img4_0.05000.txt    gt=verified     pred=verified   OK      0.06s
# [dbg] fc7/img0_0.50000.txt [pgd] restart=0 step=2: found adversarial example (pred=9)
# [dbg] fc7/img0_0.50000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# fc7/img0_0.50000.txt    gt=not verified pred=not verified       OK      0.00s
# [dbg] fc7/img1_0.14500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img1_0.14500.txt [deeppoly] unstable per layer=[11, 9, 10, 9, 12] total=51
# [dbg] fc7/img1_0.14500.txt [deeppoly] 0/1-heuristic min_margin=1.8369 already_certified=9/9
# [dbg] fc7/img1_0.14500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc7/img1_0.14500.txt total_time=0.07s result=VERIFIED
# fc7/img1_0.14500.txt    gt=verified     pred=verified   OK      0.07s
# [dbg] fc7/img2_0.17000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img2_0.17000.txt [deeppoly] unstable per layer=[23, 13, 9, 8, 4] total=57
# [dbg] fc7/img2_0.17000.txt [deeppoly] 0/1-heuristic min_margin=0.0384 already_certified=9/9
# [dbg] fc7/img2_0.17000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc7/img2_0.17000.txt total_time=0.07s result=VERIFIED
# fc7/img2_0.17000.txt    gt=verified     pred=verified   OK      0.07s
# [dbg] fc7/img3_0.02000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img3_0.02000.txt [deeppoly] unstable per layer=[6, 1, 5, 4, 5] total=21
# [dbg] fc7/img3_0.02000.txt [deeppoly] 0/1-heuristic min_margin=1.5143 already_certified=9/9
# [dbg] fc7/img3_0.02000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc7/img3_0.02000.txt total_time=0.07s result=VERIFIED
# fc7/img3_0.02000.txt    gt=verified     pred=verified   OK      0.07s
# [dbg] fc7/img4_0.15500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] fc7/img4_0.15500.txt [deeppoly] unstable per layer=[13, 7, 11, 12, 4] total=47
# [dbg] fc7/img4_0.15500.txt [deeppoly] 0/1-heuristic min_margin=0.1385 already_certified=9/9
# [dbg] fc7/img4_0.15500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.01s)
# [dbg] fc7/img4_0.15500.txt total_time=0.07s result=VERIFIED
# fc7/img4_0.15500.txt    gt=verified     pred=verified   OK      0.07s
# [dbg] conv1/img0_0.13500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img0_0.13500.txt [deeppoly] unstable per layer=[2082, 44] total=2126
# [dbg] conv1/img0_0.13500.txt [deeppoly] 0/1-heuristic min_margin=1.1042 already_certified=9/9
# [dbg] conv1/img0_0.13500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.16s)
# [dbg] conv1/img0_0.13500.txt total_time=0.24s result=VERIFIED
# conv1/img0_0.13500.txt  gt=verified     pred=verified   OK      0.24s
# [dbg] conv1/img1_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img1_0.13000.txt [deeppoly] unstable per layer=[1964, 38] total=2002
# [dbg] conv1/img1_0.13000.txt [deeppoly] 0/1-heuristic min_margin=2.0564 already_certified=9/9
# [dbg] conv1/img1_0.13000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.15s)
# [dbg] conv1/img1_0.13000.txt total_time=0.22s result=VERIFIED
# conv1/img1_0.13000.txt  gt=verified     pred=verified   OK      0.22s
# [dbg] conv1/img2_0.31000.txt [pgd] restart=0 step=3: found adversarial example (pred=9)
# [dbg] conv1/img2_0.31000.txt total_time=0.00s result=NOT VERIFIED (PGD found a counterexample)
# conv1/img2_0.31000.txt  gt=not verified pred=not verified       OK      0.00s
# [dbg] conv1/img3_0.04000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img3_0.04000.txt [deeppoly] unstable per layer=[92, 12] total=104
# [dbg] conv1/img3_0.04000.txt [deeppoly] 0/1-heuristic min_margin=8.6381 already_certified=9/9
# [dbg] conv1/img3_0.04000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.16s)
# [dbg] conv1/img3_0.04000.txt total_time=0.23s result=VERIFIED
# conv1/img3_0.04000.txt  gt=verified     pred=verified   OK      0.23s
# [dbg] conv1/img4_0.03000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv1/img4_0.03000.txt [deeppoly] unstable per layer=[101, 9] total=110
# [dbg] conv1/img4_0.03000.txt [deeppoly] 0/1-heuristic min_margin=8.7274 already_certified=9/9
# [dbg] conv1/img4_0.03000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.15s)
# [dbg] conv1/img4_0.03000.txt total_time=0.23s result=VERIFIED
# conv1/img4_0.03000.txt  gt=verified     pred=verified   OK      0.23s
# [dbg] conv2/img0_0.13000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img0_0.13000.txt [deeppoly] unstable per layer=[97, 119, 35] total=251
# [dbg] conv2/img0_0.13000.txt [deeppoly] 0/1-heuristic min_margin=-0.4597 already_certified=7/9
# [dbg] conv2/img0_0.13000.txt [deeppoly] init comparison: chosen=heuristic score=-1.2555
# [dbg] conv2/img0_0.13000.txt [deeppoly] VERIFIED during shared opt at step 3 (5.14s)
# [dbg] conv2/img0_0.13000.txt total_time=5.25s result=VERIFIED
# conv2/img0_0.13000.txt  gt=verified     pred=verified   OK      5.25s
# [dbg] conv2/img1_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img1_0.01000.txt [deeppoly] unstable per layer=[7, 12, 0] total=19
# [dbg] conv2/img1_0.01000.txt [deeppoly] 0/1-heuristic min_margin=7.5280 already_certified=9/9
# [dbg] conv2/img1_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.71s)
# [dbg] conv2/img1_0.01000.txt total_time=0.83s result=VERIFIED
# conv2/img1_0.01000.txt  gt=verified     pred=verified   OK      0.83s
# [dbg] conv2/img2_0.16500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img2_0.16500.txt [deeppoly] unstable per layer=[1240, 382, 36] total=1658
# [dbg] conv2/img2_0.16500.txt [deeppoly] 0/1-heuristic min_margin=1.1168 already_certified=9/9
# [dbg] conv2/img2_0.16500.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.69s)
# [dbg] conv2/img2_0.16500.txt total_time=0.81s result=VERIFIED
# conv2/img2_0.16500.txt  gt=verified     pred=verified   OK      0.81s
# [dbg] conv2/img3_0.27000.txt [pgd] restart=0 step=13: found adversarial example (pred=3)
# [dbg] conv2/img3_0.27000.txt total_time=0.01s result=NOT VERIFIED (PGD found a counterexample)
# conv2/img3_0.27000.txt  gt=not verified pred=not verified       OK      0.01s
# [dbg] conv2/img4_0.16000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv2/img4_0.16000.txt [deeppoly] unstable per layer=[660, 126, 32] total=818
# [dbg] conv2/img4_0.16000.txt [deeppoly] 0/1-heuristic min_margin=5.1925 already_certified=9/9
# [dbg] conv2/img4_0.16000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.71s)
# [dbg] conv2/img4_0.16000.txt total_time=0.83s result=VERIFIED
# conv2/img4_0.16000.txt  gt=verified     pred=verified   OK      0.83s
# [dbg] conv3/img0_0.01000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img0_0.01000.txt [deeppoly] unstable per layer=[9, 4, 1, 2] total=16
# [dbg] conv3/img0_0.01000.txt [deeppoly] 0/1-heuristic min_margin=9.2998 already_certified=9/9
# [dbg] conv3/img0_0.01000.txt [deeppoly] VERIFIED from the 0/1 heuristic alone (0.81s)
# [dbg] conv3/img0_0.01000.txt total_time=0.94s result=VERIFIED
# conv3/img0_0.01000.txt  gt=verified     pred=verified   OK      0.94s
# [dbg] conv3/img1_0.18500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img1_0.18500.txt [deeppoly] unstable per layer=[101, 218, 26, 44] total=389
# [dbg] conv3/img1_0.18500.txt [deeppoly] 0/1-heuristic min_margin=-2.0850 already_certified=8/9
# [dbg] conv3/img1_0.18500.txt [deeppoly] init comparison: chosen=heuristic score=-3.0266
# [dbg] conv3/img1_0.18500.txt [deeppoly] VERIFIED during shared opt at step 8 (10.78s)
# [dbg] conv3/img1_0.18500.txt total_time=10.91s result=VERIFIED
# conv3/img1_0.18500.txt  gt=verified     pred=verified   OK      10.91s
# [dbg] conv3/img2_0.23500.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img2_0.23500.txt [deeppoly] unstable per layer=[110, 188, 34, 45] total=377
# [dbg] conv3/img2_0.23500.txt [deeppoly] 0/1-heuristic min_margin=-2.0665 already_certified=7/9
# [dbg] conv3/img2_0.23500.txt [deeppoly] init comparison: chosen=heuristic score=-4.1445
# [dbg] conv3/img2_0.23500.txt [deeppoly] VERIFIED during shared opt at step 5 (8.00s)
# [dbg] conv3/img2_0.23500.txt total_time=8.14s result=VERIFIED
# conv3/img2_0.23500.txt  gt=verified     pred=verified   OK      8.14s
# [dbg] conv3/img3_0.24000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img3_0.24000.txt [deeppoly] unstable per layer=[145, 283, 32, 55] total=515
# [dbg] conv3/img3_0.24000.txt [deeppoly] 0/1-heuristic min_margin=-1.4540 already_certified=7/9
# [dbg] conv3/img3_0.24000.txt [deeppoly] init comparison: chosen=heuristic score=-3.6600
# [dbg] conv3/img3_0.24000.txt [deeppoly] VERIFIED during shared opt at step 2 (5.14s)
# [dbg] conv3/img3_0.24000.txt total_time=5.28s result=VERIFIED
# conv3/img3_0.24000.txt  gt=verified     pred=verified   OK      5.28s
# [dbg] conv3/img4_0.24000.txt [pgd] no adversarial example found after 5 restarts x 50 steps
# [dbg] conv3/img4_0.24000.txt [deeppoly] unstable per layer=[71, 133, 31, 47] total=282
# [dbg] conv3/img4_0.24000.txt [deeppoly] 0/1-heuristic min_margin=-0.3855 already_certified=8/9
# [dbg] conv3/img4_0.24000.txt [deeppoly] init comparison: chosen=heuristic score=-1.8949
# [dbg] conv3/img4_0.24000.txt [deeppoly] VERIFIED during shared opt at step 1 (4.29s)
# [dbg] conv3/img4_0.24000.txt total_time=4.42s result=VERIFIED
# conv3/img4_0.24000.txt  gt=verified     pred=verified   OK      4.42s

# ==================== SUMMARY ====================
# Overall: 46/50 = 92.0%
# Total wall time: 41.8s, avg 0.84s/case

# net      correct  total 
# conv1           5      5
# conv2           5      5
# conv3           5      5
# fc1             4      5
# fc2             5      5
# fc3             3      5
# fc4             5      5
# fc5             4      5
# fc6             5      5
# fc7             5      5

# No soundness violations (good -- verifier only ever loses points by being imprecise, never by being unsound).

# False negatives (gt=verified, we said not verified) -- 4 cases:
#   fc1/img2_0.11500.txt  (0.86s)
#   fc3/img0_0.04000.txt  (0.29s)
#   fc3/img2_0.04500.txt  (0.29s)
#   fc5/img1_0.12000.txt  (0.44s)