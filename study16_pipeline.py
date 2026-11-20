#!/usr/bin/env python3
"""Reproducible 16x16 MARL experiment runner for lha2326/MAIC-Research.

6 configurations x 3 environments x 5 seeds x 10,000 episodes.
It uses upstream original MAIC and variants/exactkl from the user's repository.
Does not resume partial checkpoints (saved .pth excludes replay buffer / optimizers).

Commands:
    python3 study16_pipeline.py prepare
    python3 study16_pipeline.py train --jobs 2
    python3 study16_pipeline.py status
    python3 study16_pipeline.py eval --jobs 2
    python3 study16_pipeline.py eval --jobs 2 --intermediate
    python3 study16_pipeline.py summarize

External incremental backup (strongly recommended):
    export STUDY16_BACKUP_REMOTE='mydrive:MAIC-Research/study16'
Requires configured rclone and destination. Each completed run is copied off-host.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / 'source_maic'
ORIG_REF = ORIGINAL / 'refactor'
EXACT_REF = ROOT / 'variants' / 'exactkl'
RUNS = ROOT / 'runs_study16'
REPORTS = ROOT / 'reports_study16'
PIN = 'ded6164f99742db4769a7f05b45f6e6007ff8b1a'
ENVS = {
    'MTE16': 'MultiGrid-MultiTargetEmpty-16x16-v0',
    'PushBox16': 'MultiGrid-PushBox-16x16-v0',
    'PassSparse16': 'MultiGrid-PassSparse-16x16-v0',
}
ALGOS = {
    'MAIC': ('original', 1.0, 1.0),
    'MASAC': ('original', 0.0, 1.0),
    'MADDPG': ('maddpg', 0.0, 0.0),
    'COMA_Phi': ('original', 0.0, 0.0),
    'MAIC_ExactKL': ('exactkl', 1.0, 1.0),
    'MASAC_ExactKL': ('exactkl', 0.0, 1.0),
}
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def jread(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def jwrite(p: Path, v):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(v, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(p)


def stamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def ensure_sources():
    if not (ORIG_REF / 'train_config.json').is_file():
        raise SystemExit('Missing original MAIC. Clone as source_maic and checkout pinned commit (see SETUP_STUDY16.md).')
    if not (EXACT_REF / 'main.py').is_file():
        raise SystemExit('Missing variants/exactkl from MAIC-Research GitHub.')
    rc = subprocess.run(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    head = rc.stdout.strip()
    if rc.returncode or head != PIN:
        raise SystemExit(f'Original MAIC commit mismatch: {head or "not a git repo"}; expected {PIN}')
    # Guard the largest confound: modified environment implementations across variants.
    for name in ('multitargetempty.py', 'push_box.py', 'pass_spare.py'):
        a = ORIG_REF / 'multigrid' / 'multigrid' / 'envs' / name
        b = EXACT_REF / 'multigrid' / 'multigrid' / 'envs' / name
        if not (a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()):
            raise SystemExit(f'Environment differs across original/ExactKL: {name}. Audit before comparing.')


def target_jobs(args):
    envs = args.envs or list(ENVS)
    algos = args.algos or list(ALGOS)
    seeds = args.seeds or DEFAULT_SEEDS
    for e in envs:
        if e not in ENVS:
            raise SystemExit('Unknown env '+e)
    for a in algos:
        if a not in ALGOS:
            raise SystemExit('Unknown algorithm '+a)
    for e in envs:
        for a in algos:
            for seed in seeds:
                yield e, a, int(seed)


def rd_for(job):
    env, algo, seed = job
    return RUNS / env / algo / f'seed{seed}'


def prepare_one(job, episodes, save_every, make=True):
    env, algo, seed = job
    rd = rd_for(job)
    runner, ig, entropy = ALGOS[algo]
    cfg = jread(ORIG_REF / 'train_config.json')
    cfg.update({
        'env_id': ENVS[env],
        'seed': seed,
        'episodes': int(episodes),
        'steps_per_episode': 40,
        'num_agents': 2,
        'obs_state_mode': 'simple',
        'scaled_information_gain_coef': ig,
        'scaled_entropy_coef': entropy,
        'model_dir': str(rd.resolve()),
        'record_video': False,  # videos can add very large disk footprint
        'render': False,
        'save_every': int(save_every),
        'plot_every': 250,
        'metrics_save_every': 10,
    })
    # Original paper settings. Don't inadvertently run COMA (IG=0, H=0)
    # instead of MAIC (IG=1,H=1).
    assert cfg['model_config']['alpha_kl'] == 1.0, 'Unexpected original alpha_kl'
    meta = {
        'env_key': env, 'env_id': ENVS[env], 'algo': algo, 'seed': seed,
        'runner': runner, 'base_source_commit': PIN,
        'episodes': int(episodes), 'steps_per_episode': 40,
        'information_gain_coef': ig, 'entropy_coef': entropy,
        'start_config_created_at_utc': stamp(),
    }
    configpath = rd / 'config.json'
    if configpath.exists():
        prior = jread(configpath)
        if prior != cfg:
            raise RuntimeError(f'Existing config differs: {configpath}. Never overwrite running/old experiment; inspect first.')
        return rd
    if any(rd.iterdir()) if rd.exists() else False:
        raise RuntimeError(f'Non-empty directory without config: {rd}. Refusing overwrite.')
    if make:
        rd.mkdir(parents=True, exist_ok=True)
        jwrite(configpath, cfg)
        jwrite(rd / 'run_meta.json', meta)
    return rd


def backup_run(rd: Path):
    remote = os.environ.get('STUDY16_BACKUP_REMOTE', '').strip()
    if not remote:
        return True, 'not configured'
    if not shutil.which('rclone'):
        return False, 'rclone missing: backup was requested but rclone is not installed'
    target = remote.rstrip('/') + '/' + rd.relative_to(RUNS).as_posix()
    cmd = ['rclone', 'copy', str(rd), target, '--transfers', '4', '--checkers', '4', '--retries', '3']
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode:
        return False, ret.stderr[-600:]
    return True, target


def train_one(job, retry_failed=False):
    rd = rd_for(job)
    cfg = rd / 'config.json'
    if not cfg.exists():
        return job, 'NO CONFIG (run prepare first)', None
    done = rd / 'DONE.json'
    if done.exists() and jread(done).get('returncode') == 0 and (rd / 'model_final.pth').is_file():
        return job, 'SKIP complete', None
    if (rd / 'train.log').exists() or done.exists() or (rd / 'metrics_latest.json').exists():
        if not retry_failed:
            return job, 'INCOMPLETE exists: preserved. Use --retry-failed to archive then restart from 0.', None
        # don't silently delete partial runs; archive with timestamp
        suffix = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        arch = ROOT / 'backups_study16_partial' / rd.relative_to(RUNS) / suffix
        arch.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(rd), str(arch))
        md = jread(arch / 'run_meta.json')
        pcfg = jread(arch / 'config.json')
        rd.mkdir(parents=True, exist_ok=True)
        jwrite(rd / 'config.json', pcfg)
        jwrite(rd / 'run_meta.json', md)
    env, algo, seed = job
    runner = ALGOS[algo][0]
    if runner == 'original':
        cmd = [sys.executable, 'main.py', '--config', str(cfg)]
        cwd = ORIG_REF
    elif runner == 'exactkl':
        cmd = [sys.executable, 'main.py', '--config', str(cfg)]
        cwd = EXACT_REF
    else:
        cmd = [sys.executable, str(ROOT / 'baselines' / 'run_maddpg.py'),
               '--maic-root', str(ORIG_REF), '--config', str(cfg)]
        cwd = ORIG_REF
    envvars = dict(os.environ)
    envvars['OMP_NUM_THREADS'] = envvars.get('OMP_NUM_THREADS', '1')
    envvars['MKL_NUM_THREADS'] = envvars.get('MKL_NUM_THREADS', '1')
    start = time.monotonic()
    print(f'[{stamp()}] START {env}/{algo}/seed{seed}', flush=True)
    with (rd / 'train.log').open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=cwd, env=envvars, stdout=f, stderr=subprocess.STDOUT)
    took = time.monotonic() - start
    final_ok = (rd / 'model_final.pth').is_file()
    status = 0 if p.returncode == 0 and final_ok else (p.returncode or 2)
    jwrite(done, {'returncode': status, 'process_returncode': p.returncode, 'seconds': round(took, 1),
                  'cmd': cmd, 'cwd': str(cwd), 'finished_at_utc': stamp(), 'final_checkpoint': final_ok})
    if status == 0:
        backed, message = backup_run(rd)
        if not backed:
            print(f'*** BACKUP FAILED {env}/{algo}/seed{seed}: {message}', flush=True)
        return job, 'OK'+('; backup '+message if backed and message != 'not configured' else ''), round(took, 1)
    return job, f'FAIL code {status}: see {rd / "train.log"}', round(took, 1)


def do_train(args):
    ensure_sources()
    jobs = list(target_jobs(args))
    for job in jobs:
        prepare_one(job, args.episodes, args.save_every)
    print(f'Training {len(jobs)} runs, {args.episodes} episodes each, up to {args.jobs} in parallel', flush=True)
    if not os.environ.get('STUDY16_BACKUP_REMOTE'):
        print('WARNING: STUDY16_BACKUP_REMOTE not configured. All raw checkpoints remain only on VAST!', flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(train_one, job, args.retry_failed) for job in jobs]
        for future in as_completed(futures):
            job, state, secs = future.result()
            print(f'[{stamp()}] {"/".join(map(str,job))}: {state}' +
                  (f' ({secs / 60:.1f} min)' if secs is not None else ''), flush=True)


def do_status(args):
    jobs = list(target_jobs(args))
    total_steps, completed, errors = 0, 0, 0
    print(f'STUDY16 | {stamp()} | {len(jobs)} jobs\n')
    for env in args.envs or ENVS:
        print(f'[{env}]')
        for algo in args.algos or ALGOS:
            pieces=[]
            for seed in args.seeds or DEFAULT_SEEDS:
                rd = rd_for((env, algo, seed))
                cfgpath = rd / 'config.json'
                if not cfgpath.is_file():
                    pieces.append(f'{seed}:---')
                    continue
                cfg = jread(cfgpath)
                cap = cfg['episodes']
                mt = rd / 'metrics_latest.json'
                n = 0
                if mt.is_file():
                    try:
                        m = jread(mt)
                        n = max(int(m.get('episode', 0)), len(m.get('episode_rewards', [])))
                    except (ValueError, json.JSONDecodeError):
                        pass
                done = rd / 'DONE.json'
                if done.is_file():
                    code = jread(done).get('returncode')
                    if code == 0 and (rd / 'model_final.pth').is_file():
                        completed += 1
                    else:
                        errors += 1
                total_steps += min(n, cap)
                pieces.append(f'{seed}:{n}/{cap}' + ('✓' if done.is_file() and jread(done).get('returncode') == 0 else ''))
            print(f'  {algo:<18} '+ ' | '.join(pieces))
    goal = sum(jread(rd_for(j) / 'config.json')['episodes'] for j in jobs if (rd_for(j)/'config.json').is_file())
    print(f'\nEPISODES {total_steps:,}/{goal:,} ({(total_steps/goal*100 if goal else 0):.2f}%)')
    print(f'DONE {completed}/{len(jobs)} | FAILED {errors} | EVALUATION: count JSON results separately')


def evaluate_one(task):
    job, ckpt, mode, episodes, evseed = task
    env, algo, seed = job
    rd = rd_for(job)
    source = EXACT_REF if ALGOS[algo][0]=='exactkl' else ORIG_REF
    out = rd / f'eval_{ckpt.removesuffix(".pth")}_{mode}.json'
    if out.is_file():
        return job, ckpt, mode, 'SKIP existing'
    cmd = [sys.executable, str(ROOT / 'tools' / 'evaluate.py'),
           '--maic-root', str(source), '--run-dir', str(rd),
           '--checkpoint', ckpt, '--episodes', str(episodes), '--mode', mode,
           '--eval-seed', str(evseed), '--out', str(out)]
    if ALGOS[algo][0]=='maddpg':
        cmd += ['--algo', 'maddpg']
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        (rd / f'eval_{ckpt.removesuffix(".pth")}_{mode}.err.log').write_text(p.stdout+'\n'+p.stderr)
        return job, ckpt, mode, 'FAIL '+p.stderr[-160:].replace('\n',' ')
    backed, msg = backup_run(rd)
    if not backed:
        print('*** EVAL BACKUP FAILED: ', msg, flush=True)
    return job, ckpt, mode, 'OK'


def do_eval(args):
    jobs = list(target_jobs(args))
    tasks=[]
    for job in jobs:
        rd=rd_for(job)
        done=rd/'DONE.json'
        if not(done.is_file() and jread(done).get('returncode')==0 and (rd/'model_final.pth').is_file()):
            continue
        algo=job[1]
        cps=['model_final.pth']
        if args.intermediate and algo in ('MAIC','MASAC','MAIC_ExactKL','MASAC_ExactKL'):
            cps += [f'model_ep{v}.pth' for v in (2000,4000,6000,8000)]
        for cp in cps:
            if not (rd/cp).exists():
                print('MISSING checkpoint', rd/cp, flush=True)
                continue
            for mode in ('greedy','stochastic'):
                tasks.append((job,cp,mode,args.eval_episodes,args.eval_seed))
    print(f'{len(tasks)} evaluations; {args.jobs} parallel', flush=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fs=[pool.submit(evaluate_one,t) for t in tasks]
        for f in as_completed(fs):
            j,cp,mo,res=f.result()
            print('/'.join(map(str,j)), cp, mo, res, flush=True)


def do_summary(args):
    REPORTS.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable,str(ROOT/'tools'/'aggregate.py'),
           '--runs',str(RUNS),'--out',str(REPORTS)]
    rc=subprocess.run(cmd,cwd=ROOT).returncode
    if rc:
        raise SystemExit(rc)
    print('Summary saved at', REPORTS)


def main():
    p=argparse.ArgumentParser(description='16x16 6-method MAIC experiment')
    p.add_argument('command', choices=['prepare','train','status','eval','summarize'])
    p.add_argument('--envs', nargs='+', choices=list(ENVS))
    p.add_argument('--algos', nargs='+', choices=list(ALGOS))
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--episodes', type=int, default=10000)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--retry-failed', action='store_true', help='Archive incomplete run then restart it from zero (NOT checkpoint resume)')
    p.add_argument('--eval-episodes', type=int, default=100)
    p.add_argument('--eval-seed', type=int, default=20260)
    p.add_argument('--intermediate', action='store_true', help='Include intermediate 2k, 4k, 6k, 8k on MAIC/MASAC and ExactKL variants')
    args=p.parse_args()
    if args.episodes<1 or args.save_every<1 or args.jobs<1:
        p.error('episodes, save_every, and jobs must be positive')
    if args.command in ('prepare','train'):
        ensure_sources()
    if args.command=='prepare':
        for job in target_jobs(args):
            prepare_one(job,args.episodes,args.save_every)
        print('Prepared run configs in',RUNS)
    elif args.command=='train': do_train(args)
    elif args.command=='status': do_status(args)
    elif args.command=='eval': do_eval(args)
    elif args.command=='summarize': do_summary(args)


if __name__=='__main__':
    main()
