"""Evaluate model_final.pth of every finished run (greedy + stochastic, fixed eval seed).

  python tools/evaluate_all.py --runs runs -j 4
"""
import argparse, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=str(HERE / 'runs'))
    p.add_argument('--exp', default=str(HERE / 'experiments.json'))
    p.add_argument('--maic-root', default=str(HERE / 'third_party' / 'MAIC' / 'refactor'))
    p.add_argument('--checkpoint', default='model_final.pth')
    p.add_argument('-j', '--jobs', type=int, default=2)
    p.add_argument('--force', action='store_true')
    a = p.parse_args()
    ev = json.loads(Path(a.exp).read_text())['eval']
    tasks = []
    for done in sorted(Path(a.runs).glob('*/*/seed*/DONE.json')):
        rd = done.parent
        if json.loads(done.read_text()).get('returncode') != 0 or not (rd / a.checkpoint).exists():
            continue
        meta = json.loads((rd / 'run_meta.json').read_text())
        for mode in ev['modes']:
            out = rd / f'eval_{Path(a.checkpoint).stem}_{mode}.json'
            if out.exists() and not a.force:
                continue
            cmd = [sys.executable, str(HERE / 'tools' / 'evaluate.py'), '--maic-root', a.maic_root, '--run-dir', str(rd),
                   '--checkpoint', a.checkpoint, '--episodes', str(ev['episodes']), '--eval-seed', str(ev['eval_seed']),
                   '--mode', mode, '--out', str(out)]
            if meta.get('runner') == 'maddpg':
                cmd += ['--algo', 'maddpg']
            tasks.append((rd, mode, cmd))
    print(len(tasks), 'evaluations')

    def go(t):
        rd, mode, cmd = t
        r = subprocess.run(cmd, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [''])[-1] if r.returncode == 0 else r.stderr[-400:]
        print(rd.relative_to(Path(a.runs)), mode, 'ok' if r.returncode == 0 else 'FAIL', tail[:200], flush=True)

    with ThreadPoolExecutor(a.jobs) as ex:
        list(ex.map(go, tasks))


if __name__ == '__main__':
    main()
