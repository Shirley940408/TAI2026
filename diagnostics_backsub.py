"""
Scoring diagnostic for the verifier.

`not verified` is the CORRECT answer for a case that is genuinely not robust,
so the raw count of `verified` lines says nothing on its own.  This script
splits every case into four buckets by cross-checking the verifier against a
much stronger attack than the one running inside it:

    VERIFIED    verifier proved it, attack found nothing        -> correct
    NON-ROBUST  verifier declined, attack found a real
                counterexample inside the eps-ball              -> correct
    MISS        verifier declined, attack found nothing         -> the only
                                                                   real losses
    UNSOUND     verifier proved it, attack broke it             -> critical bug

    score = VERIFIED + NON-ROBUST

The attack is deliberately much stronger than the verifier's internal
pre-check (many more restarts and steps, several step sizes, and a targeted
run against every wrong class), so a case landing in MISS rather than
NON-ROBUST is good evidence that it really is robust and really was missed.
Note the asymmetry: an attack failing never proves robustness, so MISS is
"probably a genuine miss", while NON-ROBUST is certain -- a counterexample is
a proof.

Also reports per-case wall time, since the project has a per-case limit and a
verifier that times out on the grading machine loses the case.

Usage:
    python diagnostics_backsub.py --tests-dir ../test_cases
    python diagnostics_backsub.py --tests-dir ../test_cases --verifier backsub_verifier
    python diagnostics_backsub.py --tests-dir ../test_cases --net fc3
"""

import argparse
import importlib
import os
import time

import torch

DEVICE = 'cpu'
INPUT_SIZE = 28
NETWORK_NAMES = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7',
                 'conv1', 'conv2', 'conv3']


# ---------------------------------------------------------------------------
# a deliberately strong attack, used only to classify the verifier's output
# ---------------------------------------------------------------------------

def strong_attack(model, x, eps, true_label, restarts=20, steps=250, seed=0):
    """Search hard for a counterexample. Returns (found, wrong_label, x_adv)."""
    torch.manual_seed(seed)
    lower = torch.clamp(x.detach() - eps, 0.0, 1.0)
    upper = torch.clamp(x.detach() + eps, 0.0, 1.0)
    n_class = model(x).shape[1]
    targets = [c for c in range(n_class) if c != true_label]

    def check(point):
        with torch.no_grad():
            pred = int(model(point)[0].argmax())
        return pred != true_label, pred

    ok, pred = check(x)
    if ok:
        return True, pred, x.detach()

    # A few step sizes: a single schedule can stall on flat regions.
    schedules = [eps / 4.0, eps / 10.0, eps / 40.0]

    def run(loss_fn, n_restarts, n_steps):
        for r in range(n_restarts):
            if r == 0:
                adv = x.detach().clone()
            else:
                adv = lower + (upper - lower) * torch.rand_like(x)
            step = schedules[r % len(schedules)]
            for it in range(n_steps):
                adv = adv.detach().requires_grad_(True)
                logits = model(adv)[0]
                if int(logits.argmax()) != true_label:
                    return True, int(logits.argmax()), adv.detach()
                loss = loss_fn(logits)
                grad = torch.autograd.grad(loss, adv)[0]
                # decay the step so late iterations can settle into a corner
                cur = step * (0.1 ** (it / max(n_steps - 1, 1)))
                adv = adv.detach() + cur * grad.sign()
                adv = torch.max(torch.min(adv, upper), lower)
            found, pred = check(adv)
            if found:
                return True, pred, adv.detach()
        return False, None, None

    # untargeted: push the best wrong class up
    def untargeted(logits):
        other = torch.cat([logits[:true_label], logits[true_label + 1:]])
        return other.max() - logits[true_label]

    found, pred, adv = run(untargeted, restarts, steps)
    if found:
        return True, pred, adv

    # targeted: one dedicated run per wrong class
    per_target = max(2, restarts // len(targets))
    for target in targets:
        def targeted(logits, t=target):
            return logits[t] - logits[true_label]
        found, pred, adv = run(targeted, per_target, steps)
        if found:
            return True, pred, adv

    return False, None, None


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests-dir', default='../test_cases')
    parser.add_argument('--verifier', default='verifier',
                        help='module to import analyze/load_network/parse_spec from')
    parser.add_argument('--net', choices=NETWORK_NAMES,
                        help='restrict to one network')
    parser.add_argument('--time-limit', type=float, default=180.0,
                        help='per-case limit to flag against')
    parser.add_argument('--restarts', type=int, default=20)
    parser.add_argument('--steps', type=int, default=250)
    parser.add_argument('--skip-attack', action='store_true',
                        help='only time the verifier, do not classify')
    args = parser.parse_args()

    V = importlib.import_module(args.verifier)
    base = os.path.abspath(args.tests_dir)
    nets = [args.net] if args.net else NETWORK_NAMES

    rows = []
    print(f'{"net":<8}{"case":<22}{"verifier":>14}{"attack":>10}'
          f'{"verdict":>13}{"time":>9}')
    print('-' * 76)

    for net_name in nets:
        net_dir = os.path.join(base, net_name)
        if not os.path.isdir(net_dir):
            continue
        net = V.load_network(net_name)
        net.eval()

        for fn in sorted(f for f in os.listdir(net_dir) if f.endswith('.txt')):
            path = os.path.join(net_dir, fn)
            true_label, pixels, eps = V.parse_spec(path)
            xs = torch.FloatTensor(pixels).view(
                1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
            if int(net(xs).max(dim=1)[1]) != true_label:
                print(f'{net_name:<8}{fn:<22}{"BAD SPEC":>14}')
                continue

            t0 = time.perf_counter()
            try:
                proved = bool(V.analyze(net, xs, eps, true_label))
                err = None
            except Exception as exc:
                proved, err = False, repr(exc)
            elapsed = time.perf_counter() - t0

            if args.skip_attack:
                attacked = None
            else:
                attacked, _, _ = strong_attack(
                    net, xs, eps, true_label, args.restarts, args.steps)

            if err is not None:
                verdict = 'ERROR'
            elif attacked is None:
                verdict = 'verified' if proved else 'not verified'
            elif proved and attacked:
                verdict = 'UNSOUND'
            elif proved:
                verdict = 'VERIFIED'
            elif attacked:
                verdict = 'NON-ROBUST'
            else:
                verdict = 'MISS'

            rows.append(dict(net=net_name, case=fn, eps=eps, proved=proved,
                             attacked=attacked, verdict=verdict,
                             time=elapsed, err=err))
            flag = '  <-- OVER LIMIT' if elapsed > args.time_limit else ''
            print(f'{net_name:<8}{fn:<22}'
                  f'{("proved" if proved else "declined"):>14}'
                  f'{("broken" if attacked else ("-" if attacked is not None else "n/a")):>10}'
                  f'{verdict:>13}{elapsed:>8.1f}s{flag}', flush=True)
            if err:
                print(f'         exception: {err}')

    # ---------------- summary ----------------
    print()
    counts = {}
    for r in rows:
        counts[r['verdict']] = counts.get(r['verdict'], 0) + 1
    total = len(rows)
    correct = counts.get('VERIFIED', 0) + counts.get('NON-ROBUST', 0)

    print('=' * 76)
    print(f'{"VERIFIED   (proved robust)":<44}{counts.get("VERIFIED", 0):>4}')
    print(f'{"NON-ROBUST (counterexample found, correct answer)":<44}'
          f'{counts.get("NON-ROBUST", 0):>4}')
    print(f'{"MISS       (probably robust, not proved)":<44}{counts.get("MISS", 0):>4}')
    print(f'{"UNSOUND    (proved but attack broke it)":<44}{counts.get("UNSOUND", 0):>4}')
    print(f'{"ERROR":<44}{counts.get("ERROR", 0):>4}')
    print('-' * 76)
    print(f'{"SCORE":<44}{correct:>4} / {total}'
          f'   ({100.0*correct/max(total,1):.0f}%)')
    print('=' * 76)

    if counts.get('UNSOUND'):
        print()
        print('!!! SOUNDNESS VIOLATIONS -- these must be fixed before anything else:')
        for r in rows:
            if r['verdict'] == 'UNSOUND':
                print(f'    {r["net"]}/{r["case"]}  eps={r["eps"]}')

    misses = [r for r in rows if r['verdict'] == 'MISS']
    if misses:
        print()
        print('remaining misses (the only cases worth more work):')
        for r in misses:
            print(f'    {r["net"]:<8}{r["case"]:<22}eps={r["eps"]:<8}'
                  f'{r["time"]:.1f}s')

    if rows:
        print()
        times = sorted((r['time'] for r in rows), reverse=True)
        print(f'time: total {sum(times):.0f}s   median {times[len(times)//2]:.1f}s'
              f'   slowest {times[0]:.1f}s')
        slow = [r for r in rows if r['time'] > args.time_limit]
        if slow:
            print(f'  {len(slow)} case(s) over the {args.time_limit:.0f}s limit '
                  f'-- these are losses on a slower grading machine:')
            for r in slow:
                print(f'    {r["net"]}/{r["case"]}  {r["time"]:.1f}s  ({r["verdict"]})')
        near = [r for r in rows
                if 0.7 * args.time_limit < r['time'] <= args.time_limit]
        if near:
            print(f'  {len(near)} case(s) within 30% of the limit:')
            for r in near:
                print(f'    {r["net"]}/{r["case"]}  {r["time"]:.1f}s  ({r["verdict"]})')


if __name__ == '__main__':
    main()