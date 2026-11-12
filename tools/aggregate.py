"""Collect all runs into tables and figures.

Outputs (in --out, default reports/):
  runs.csv           one row per run (training + evaluation numbers)
  summary.csv        mean +- std over seeds per (env, algo)
  summary.md         the same as a Markdown table
  curves_<env>.png   training reward and unique states seen (mean +- std over seeds)

  python tools/aggregate.py --runs runs --out reports
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def smooth(x, w):
    x = np.asarray(x, float)
    if len(x) < w: return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    return np.concatenate([np.full(w - 1, np.nan), (c[w:] - c[:-w]) / w])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='runs'); p.add_argument('--out', default='reports')
    p.add_argument('--window', type=int, default=100, help='smoothing window (episodes)')
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], defaultdict(lambda: defaultdict(list))
    for mpath in sorted(Path(a.runs).glob('*/*/seed*/metrics_latest.json')):
        rd = mpath.parent
        meta = json.loads((rd / 'run_meta.json').read_text()) if (rd / 'run_meta.json').exists() else {}
        env, algo, seed = rd.parts[-3], rd.parts[-2], rd.name.replace('seed', '')
        m = json.loads(mpath.read_text())
        r = np.asarray(m['episode_rewards'], float); n = len(r); tail = r[-max(1, n // 10):]
        us = [u for u in m.get('unique_states_seen', []) if u is not None]
        done = json.loads((rd / 'DONE.json').read_text()) if (rd / 'DONE.json').exists() else {}
        row = dict(env=env, algo=algo, seed=seed, episodes=n, finished=done.get('returncode') == 0,
                   minutes=round(done.get('seconds', 0) / 60, 1),
                   train_reward_last10pct=round(float(tail.mean()), 3),
                   train_reward_max_window=round(float(np.nanmax(smooth(r, min(a.window, n)))), 3) if n else None,
                   unique_states_final=us[-1] if us else None)
        for mode in ('greedy', 'stochastic'):
            ep = rd / f'eval_model_final_{mode}.json'
            if ep.exists():
                e = json.loads(ep.read_text())
                row[f'{mode}_reward'] = round(e['reward_mean'], 3); row[f'{mode}_success'] = round(e['success_rate'], 3)
                row[f'{mode}_agent_pos_rate'] = '/'.join(f'{x:.2f}' for x in e['per_agent_positive_rate'])
        rows.append(row)
        curves[env][algo].append((r, np.asarray([u if u is not None else np.nan for u in m.get('unique_states_seen', [])], float)))
    if not rows:
        print('no runs found'); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'runs.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)

    num = ['train_reward_last10pct', 'unique_states_final', 'greedy_reward', 'greedy_success',
           'stochastic_reward', 'stochastic_success']
    groups = defaultdict(list)
    for r in rows: groups[(r['env'], r['algo'])].append(r)
    summ = []
    for (env, algo), rs in sorted(groups.items()):
        s = dict(env=env, algo=algo, n_seeds=len(rs))
        for k in num:
            v = [r[k] for r in rs if r.get(k) is not None]
            s[k] = f'{np.mean(v):.3f} ± {np.std(v):.3f}' if v else ''
        summ.append(s)
    with open(out / 'summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, list(summ[0])); w.writeheader(); w.writerows(summ)
    with open(out / 'summary.md', 'w') as f:
        cols = list(summ[0]); f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
        for s in summ: f.write('| ' + ' | '.join(str(s[c]) for c in cols) + ' |\n')
    print((out / 'summary.md').read_text())

    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    for env, by_algo in curves.items():
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        for algo, runs in sorted(by_algo.items()):
            L = min(len(x[0]) for x in runs); w = min(a.window, L)
            R = np.stack([smooth(x[0][:L], w) for x in runs]); U = np.stack([x[1][:L] for x in runs])
            xs = np.arange(L)
            for k, (M, axx) in enumerate([(R, ax[0]), (U, ax[1])]):
                mu, sd = np.nanmean(M, 0), np.nanstd(M, 0)
                axx.plot(xs, mu, label=f'{algo} (n={len(runs)})'); axx.fill_between(xs, mu - sd, mu + sd, alpha=.2)
        ax[0].set(title=f'{env}: training episode reward (moving avg {a.window})', xlabel='episode')
        ax[1].set(title=f'{env}: unique states seen', xlabel='episode')
        for axx in ax: axx.grid(alpha=.3); axx.legend()
        fig.tight_layout(); fig.savefig(out / f'curves_{env}.png', dpi=120); plt.close(fig)


if __name__ == '__main__':
    main()
