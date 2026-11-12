"""Run the experiment matrix (env x algo x seed) on the ORIGINAL MAIC code.

Faithfulness rules
  * third_party/MAIC must be at the pinned commit and its .py/.json files unmodified
    (checked before anything runs; use --allow-dirty only for debugging).
  * Each run's config = the repo's own refactor/train_config.json, with ONLY these keys replaced:
    env_id, seed, scaled_information_gain_coef, scaled_entropy_coef, model_dir.
    (--episodes is available for smoke tests; it is recorded in the run's config.json.)
  * MAIC/MASAC/COMA runs execute `python main.py --config <run>/config.json` inside MAIC/refactor.
  * MADDPG runs execute baselines/run_maddpg.py, which calls the repo's train.run_training
    with the agent class swapped; the loop, seeding, buffer, metrics and plots are the repo's.

Examples
  python tools/run_matrix.py --list
  python tools/run_matrix.py -j 4                       # everything in experiments.json
  python tools/run_matrix.py --envs MTE16 --algos MAIC MASAC --seeds 42 -j 2
  python tools/run_matrix.py --episodes 30 --out runs_smoke -j 2   # smoke test
"""
import argparse, itertools, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REPLACED_KEYS = ('env_id', 'seed', 'scaled_information_gain_coef', 'scaled_entropy_coef', 'model_dir')


def check_pristine(maic_dir: Path, commit: str):
    head = subprocess.run(['git', '-C', str(maic_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    if head != commit:
        return f'third_party/MAIC is at {head or "?"}, expected {commit}'
    st = subprocess.run(['git', '-C', str(maic_dir), 'status', '--porcelain', '--untracked-files=all', '--',
                         '*.py', '*.json', '*.cfg', '*.toml'], capture_output=True, text=True).stdout
    st = '\n'.join(l for l in st.splitlines() if '/runs' not in l)
    if st.strip():
        return 'third_party/MAIC has modified/untracked source files:\n' + st
    return None


def build_jobs(exp, a):
    envs = a.envs or list(exp['envs'])
    algos = a.algos or list(exp['algos'])
    seeds = a.seeds or exp['seeds']
    return [(e, al, s) for e, al, s in itertools.product(envs, algos, seeds)]


def run_one(job, exp, base_cfg, a, maic_ref: Path):
    env_key, algo, seed = job
    spec = exp['algos'][algo]
    run_dir = (Path(a.out).resolve() / env_key / algo / f'seed{seed}')
    run_dir.mkdir(parents=True, exist_ok=True)
    done = run_dir / 'DONE.json'
    if done.exists() and json.loads(done.read_text()).get('returncode') == 0 and not a.force:
        return job, 'skip', 0.0
    cfg = json.loads(json.dumps(base_cfg))
    cfg.update(env_id=exp['envs'][env_key], seed=int(seed),
               scaled_information_gain_coef=float(spec['scaled_information_gain_coef']),
               scaled_entropy_coef=float(spec['scaled_entropy_coef']),
               model_dir=str(run_dir))
    if a.episodes:
        cfg['episodes'] = int(a.episodes)
    (run_dir / 'config.json').write_text(json.dumps(cfg, indent=2))
    meta = dict(env_key=env_key, algo=algo, seed=seed, runner=spec['runner'], maic_commit=exp['maic_commit'],
                replaced_keys=list(REPLACED_KEYS) + (['episodes'] if a.episodes else []))
    (run_dir / 'run_meta.json').write_text(json.dumps(meta, indent=2))
    if spec['runner'] == 'maic':
        cmd = [sys.executable, 'main.py', '--config', str(run_dir / 'config.json')]
    elif spec['runner'] == 'maddpg':
        cmd = [sys.executable, str(HERE / 'baselines' / 'run_maddpg.py'), '--maic-root', str(maic_ref),
               '--config', str(run_dir / 'config.json')]
    else:
        raise ValueError(spec['runner'])
    envv = dict(os.environ)
    envv.setdefault('OMP_NUM_THREADS', str(a.threads))
    envv.setdefault('MKL_NUM_THREADS', str(a.threads))
    t0 = time.time()
    with open(run_dir / 'train.log', 'w') as log:
        rc = subprocess.run(cmd, cwd=str(maic_ref), stdout=log, stderr=subprocess.STDOUT, env=envv).returncode
    el = time.time() - t0
    done.write_text(json.dumps(dict(returncode=rc, seconds=round(el, 1), cmd=cmd), indent=2))
    return job, ('ok' if rc == 0 else f'FAIL rc={rc}'), el


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp', default=str(HERE / 'experiments.json'))
    p.add_argument('--maic-dir', default=str(HERE / 'third_party' / 'MAIC'))
    p.add_argument('--out', default=str(HERE / 'runs'))
    p.add_argument('--envs', nargs='*'); p.add_argument('--algos', nargs='*')
    p.add_argument('--seeds', nargs='*', type=int)
    p.add_argument('--episodes', type=int, default=None, help='override (smoke tests only)')
    p.add_argument('-j', '--jobs', type=int, default=1)
    p.add_argument('--threads', type=int, default=1, help='OMP/MKL threads per run')
    p.add_argument('--force', action='store_true'); p.add_argument('--list', action='store_true')
    p.add_argument('--allow-dirty', action='store_true')
    a = p.parse_args()

    exp = json.loads(Path(a.exp).read_text())
    maic_dir = Path(a.maic_dir).resolve(); maic_ref = maic_dir / 'refactor'
    jobs = build_jobs(exp, a)
    if a.list:
        for j in jobs: print(*j)
        print(len(jobs), 'runs'); return
    err = check_pristine(maic_dir, exp['maic_commit'])
    if err and not a.allow_dirty:
        sys.exit('[faithfulness check failed] ' + err)
    base_cfg = json.loads((maic_ref / exp['base_config']).read_text())
    print(f'{len(jobs)} runs, {a.jobs} parallel, out={a.out}', flush=True)
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(run_one, j, exp, base_cfg, a, maic_ref) for j in jobs]
        for f in as_completed(futs):
            job, status, el = f.result()
            print(f'[{time.strftime("%H:%M:%S")}] {"/".join(map(str, job))}: {status} ({el/60:.1f} min)', flush=True)


if __name__ == '__main__':
    main()
