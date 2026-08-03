#!/usr/bin/env python
"""Launch parallel FastRelax + InterfaceAnalyzer jobs with caching and crash isolation.

Each complex runs in its own Python subprocess so a PyRosetta segfault cannot
kill the pool. Successful scores and relaxed PDBs are cached; failures are
logged to JSONL for later retry.

Example:
  python scripts/launch_fastrelax.py \
      --chothia_dir data/all_structures/chothia \
      --summary_path data/sabdab_summary_all.tsv \
      --out_dir data/fastrelax_cache \
      --workers 32

  # Retry only previous failures:
  python scripts/launch_fastrelax.py --out_dir data/fastrelax_cache --retry_failures --workers 32

  # Aggregate energies pickle when done (also written periodically):
  python scripts/launch_fastrelax.py --out_dir data/fastrelax_cache --aggregate_only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import signal
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}

WORKER = Path(__file__).resolve().parent / 'fastrelax_worker.py'


def _nan_to_empty(val) -> str:
    if val is None:
        return ''
    if isinstance(val, float) and val != val:  # NaN
        return ''
    s = str(val).strip()
    if s.lower() in ('', 'nan', 'none', 'na'):
        return ''
    return s


def build_entries(
    summary_path: str,
    chothia_dir: str,
    structure_name: str = 'pdbcode',
) -> List[dict]:
    """Build FastRelax jobs.

    structure_name:
      - 'pdbcode': load ``{pdb}.pdb`` (raw SAbDab Chothia ASU).
      - 'entry_id': load ``{entry_id}.pdb`` (pre-extracted single complexes
        from ``scripts/extract_sabdab_complexes.py``).
    """
    entries = []
    with open(summary_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            ag_type = _nan_to_empty(row.get('antigen_type'))
            if ag_type not in ALLOWED_AG_TYPES:
                continue
            try:
                resolution = float(str(row['resolution']).split(',')[0])
            except Exception:
                continue
            if resolution >= 4.0:
                continue

            H = _nan_to_empty(row.get('Hchain'))
            L = _nan_to_empty(row.get('Lchain'))
            # Keep '|' for InterfaceAnalyzer multi-chain antigens, but build a
            # filesystem-safe entry id (DiffAb joins antigen chains with '').
            Ag_raw = _nan_to_empty(row.get('antigen_chain')).replace(' ', '')
            if not H and not L:
                continue
            if not Ag_raw:
                continue

            pdb = str(row['pdb']).lower()
            Ag_id = Ag_raw.replace('|', '')
            entry_id = f'{pdb}_{H}_{L}_{Ag_id}'
            stem = entry_id if structure_name == 'entry_id' else pdb
            in_pdb = os.path.join(chothia_dir, f'{stem}.pdb')
            if not os.path.isfile(in_pdb):
                continue

            entries.append({
                'id': entry_id,
                'pdbcode': pdb,
                'H': H,
                'L': L,
                'Ag': Ag_raw,
                'in_pdb': in_pdb,
                'nbytes': os.path.getsize(in_pdb),
                'structure_stem': stem,
            })
    # Small complexes first: faster early progress, fewer early timeouts.
    entries.sort(key=lambda e: (e['nbytes'], e['id']))
    return entries


def is_cached_ok(score_json: Path) -> bool:
    if not score_json.is_file():
        return False
    try:
        with open(score_json) as f:
            data = json.load(f)
        return data.get('status') == 'ok' and data.get('dG_separated') is not None
    except Exception:
        return False


def run_one(job: dict) -> dict:
    """Run one worker subprocess. Safe against non-zero exits / timeouts."""
    python_exe = job.get('python_exe') or sys.executable
    cmd = [
        python_exe, '-u', str(WORKER),
        '--entry_id', job['id'],
        '--pdbcode', job['pdbcode'],
        '--H', job['H'],
        '--L', job['L'],
        '--Ag', job['Ag'],
        '--in_pdb', job['in_pdb'],
        '--relaxed_pdb', job['relaxed_pdb'],
        '--score_json', job['score_json'],
        '--relax_repeats', str(job['relax_repeats']),
    ]
    if job.get('force_relax'):
        cmd.append('--force_relax')

    log_path = job['log_path']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    meta = {
        'entry_id': job['id'],
        'pdbcode': job['pdbcode'],
        'status': 'failed',
        'returncode': None,
        'error': None,
        'error_type': None,
        'dG_separated': None,
        'E_total': None,
        't_wall_s': None,
        'log_path': log_path,
    }

    try:
        with open(log_path, 'w') as logf:
            logf.write('CMD: ' + ' '.join(cmd) + '\n\n')
            logf.flush()
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=job['timeout_s'],
            )
            logf.write(proc.stdout or '')
            meta['returncode'] = proc.returncode

        # Prefer structured JSON from score file; fall back to last stdout line.
        if os.path.isfile(job['score_json']):
            with open(job['score_json']) as f:
                payload = json.load(f)
            meta.update({
                'status': payload.get('status', meta['status']),
                'error': payload.get('error'),
                'error_type': payload.get('error_type'),
                'dG_separated': payload.get('dG_separated'),
                'E_total': payload.get('E_total'),
                'interface': payload.get('interface'),
                'used_cached_relax': payload.get('used_cached_relax'),
            })
        elif proc.stdout:
            for line in reversed(proc.stdout.splitlines()):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        payload = json.loads(line)
                        meta['status'] = payload.get('status', meta['status'])
                        meta['error'] = payload.get('error')
                        meta['dG_separated'] = payload.get('dG_separated')
                    except json.JSONDecodeError:
                        pass
                    break

        if proc.returncode < 0:
            # Negative returncode ⇒ killed by signal (e.g. -11 SIGSEGV)
            sig = -proc.returncode
            meta['status'] = 'crashed'
            meta['error_type'] = f'signal_{sig}'
            meta['error'] = f'worker killed by signal {sig} ({signal.Signals(sig).name if sig in signal.Signals._value2member_map_ else "?"})'
            # Ensure a score json exists for caching/retry bookkeeping
            crash_json = {
                'entry_id': job['id'],
                'pdbcode': job['pdbcode'],
                'status': 'crashed',
                'error': meta['error'],
                'error_type': meta['error_type'],
                'dG_separated': None,
            }
            os.makedirs(os.path.dirname(job['score_json']), exist_ok=True)
            with open(job['score_json'], 'w') as f:
                json.dump(crash_json, f, indent=2)
                f.write('\n')

    except subprocess.TimeoutExpired as e:
        meta['status'] = 'timeout'
        meta['error_type'] = 'TimeoutExpired'
        meta['error'] = f'timed out after {job["timeout_s"]}s'
        with open(log_path, 'a') as logf:
            logf.write(f'\nTIMEOUT after {job["timeout_s"]}s\n')
            if e.stdout:
                logf.write(e.stdout if isinstance(e.stdout, str) else e.stdout.decode('utf-8', 'replace'))
        timeout_json = {
            'entry_id': job['id'],
            'pdbcode': job['pdbcode'],
            'status': 'timeout',
            'error': meta['error'],
            'error_type': 'TimeoutExpired',
            'dG_separated': None,
        }
        os.makedirs(os.path.dirname(job['score_json']), exist_ok=True)
        with open(job['score_json'], 'w') as f:
            json.dump(timeout_json, f, indent=2)
            f.write('\n')
    except Exception as e:
        meta['status'] = 'failed'
        meta['error_type'] = type(e).__name__
        meta['error'] = str(e)

    meta['t_wall_s'] = time.time() - t0
    return meta


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')


def aggregate_energies(scores_dir: Path, out_pkl: Path, key_mode: str = 'pdbcode') -> dict:
    """Build precomputed_energies.pkl from successful score JSONs.

    key_mode='pdbcode': keep the most negative (best) ΔG per pdb code.
    key_mode='id': one entry per complex id.
    """
    energies: Dict[str, float] = {}
    n_ok = 0
    for path in sorted(scores_dir.glob('*.json')):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get('status') != 'ok' or data.get('dG_separated') is None:
            continue
        n_ok += 1
        dg = float(data['dG_separated'])
        key = data['entry_id'] if key_mode == 'id' else data['pdbcode']
        if key not in energies or dg < energies[key]:
            energies[key] = dg

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, 'wb') as f:
        pickle.dump(energies, f)

    if energies:
        vals = sorted(energies.values())
        median = vals[len(vals) // 2]
        frac = sum(1 for v in vals if -100 <= v <= 0) / len(vals)
        summary = {
            'n_ok_scores': n_ok,
            'n_keys': len(energies),
            'min': vals[0],
            'median': median,
            'max': vals[-1],
            'frac_in_[-100,0]': frac,
            'out_pkl': str(out_pkl),
        }
    else:
        summary = {'n_ok_scores': 0, 'n_keys': 0, 'out_pkl': str(out_pkl)}
    return summary


def load_failure_ids(failures_jsonl: Path) -> List[str]:
    ids = []
    if not failures_jsonl.is_file():
        return ids
    with open(failures_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.append(json.loads(line)['entry_id'])
            except Exception:
                continue
    # unique, preserve order
    seen = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chothia_dir', default='data/all_structures/chothia')
    ap.add_argument('--summary_path', default='data/sabdab_summary_all.tsv')
    ap.add_argument('--out_dir', default='data/fastrelax_cache')
    ap.add_argument(
        '--structure_name',
        choices=['pdbcode', 'entry_id'],
        default='pdbcode',
        help="Input PDB filename stem. Use 'entry_id' with data/chothia_complexes "
             "(from extract_sabdab_complexes.py).",
    )
    ap.add_argument('--workers', type=int, default=32)
    ap.add_argument('--relax_repeats', type=int, default=1)
    ap.add_argument('--timeout_s', type=int, default=3600, help='Per-complex timeout (default 60 min)')
    ap.add_argument('--force', action='store_true', help='Re-run even if score cache is ok')
    ap.add_argument('--force_relax', action='store_true', help='Re-run FastRelax even if relaxed pdb exists')
    ap.add_argument('--retry_failures', action='store_true', help='Only retry entry_ids in failures.jsonl')
    ap.add_argument('--limit', type=int, default=0, help='Optional cap on number of jobs (0 = all)')
    ap.add_argument('--max_pdb_bytes', type=int, default=0,
                    help='Skip input PDBs larger than this (0 = no limit). Useful for smoke tests.')
    ap.add_argument('--aggregate_only', action='store_true')
    ap.add_argument('--energy_key', choices=['pdbcode', 'id'], default='pdbcode')
    ap.add_argument('--energies_out', default='', help='Default: <out_dir>/precomputed_energies.pkl')
    ap.add_argument('--python', default='',
                    help='Python interpreter for workers (must have pyrosetta). '
                         'Default: current sys.executable.')
    ap.add_argument('--clear_failed_scores', action='store_true',
                    help='Delete non-ok score JSONs before queueing (clean retry).')
    args = ap.parse_args()

    python_exe = args.python or sys.executable
    # Fail fast if this interpreter cannot import pyrosetta (the previous full
    # run died instantly with ModuleNotFoundError in every worker).
    probe = subprocess.run(
        [python_exe, '-c', 'import pyrosetta; print(pyrosetta.__file__)'],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        msg = (probe.stderr or probe.stdout or '').strip()
        raise SystemExit(
            f'PyRosetta not importable with:\n  {python_exe}\n{msg}\n\n'
            f'Activate the env that has pyrosetta, e.g.:\n'
            f'  conda activate diffab\n'
            f'  which python   # should be .../envs/diffab/bin/python\n'
            f'  python scripts/launch_fastrelax.py --workers 32 ...\n'
            f'Or pass it explicitly:\n'
            f'  python scripts/launch_fastrelax.py --python /path/to/envs/diffab/bin/python ...'
        )
    print(f'Worker python: {python_exe}')
    print(f'PyRosetta: {probe.stdout.strip()}')

    out_dir = Path(args.out_dir)
    relaxed_dir = out_dir / 'relaxed'
    scores_dir = out_dir / 'scores'
    logs_dir = out_dir / 'logs'
    for d in (relaxed_dir, scores_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    energies_out = Path(args.energies_out) if args.energies_out else out_dir / 'precomputed_energies.pkl'
    failures_path = out_dir / 'failures.jsonl'
    successes_path = out_dir / 'successes.jsonl'
    progress_path = out_dir / 'progress.jsonl'

    if args.aggregate_only:
        summary = aggregate_energies(scores_dir, energies_out, key_mode=args.energy_key)
        print(json.dumps(summary, indent=2))
        return

    if not WORKER.is_file():
        raise FileNotFoundError(f'worker script missing: {WORKER}')

    entries = build_entries(
        args.summary_path, args.chothia_dir, structure_name=args.structure_name,
    )
    print(f'structure_name={args.structure_name} | chothia_dir={args.chothia_dir}')
    if args.max_pdb_bytes and args.max_pdb_bytes > 0:
        before = len(entries)
        entries = [e for e in entries if e['nbytes'] <= args.max_pdb_bytes]
        print(f'Entries with antigen + pdb on disk: {before} -> {len(entries)} after max_pdb_bytes={args.max_pdb_bytes}')
    else:
        print(f'Entries with antigen + pdb on disk: {len(entries)}')

    if args.retry_failures:
        fail_ids = set(load_failure_ids(failures_path))
        # Also include score jsons that are not ok
        for p in scores_dir.glob('*.json'):
            try:
                with open(p) as f:
                    data = json.load(f)
                if data.get('status') != 'ok':
                    fail_ids.add(data.get('entry_id') or p.stem)
            except Exception:
                fail_ids.add(p.stem)
        entries = [e for e in entries if e['id'] in fail_ids]
        print(f'Retry set: {len(entries)} entries')
        # For retries, remove failed score markers so worker rewrites them
        # (keep relaxed pdb cache unless --force_relax)

    if args.clear_failed_scores:
        n_cleared = 0
        for p in scores_dir.glob('*.json'):
            if not is_cached_ok(p):
                p.unlink(missing_ok=True)
                n_cleared += 1
        print(f'Cleared {n_cleared} non-ok score JSON files')

    jobs = []
    n_skip = 0
    for e in entries:
        score_json = scores_dir / f"{e['id']}.json"
        if (not args.force) and is_cached_ok(score_json):
            n_skip += 1
            continue
        if args.retry_failures:
            # only queue ids that previously failed / have no ok cache
            pass
        # Cache relaxed structures per structure_stem (entry_id when using
        # extracted complexes) so multi-copy ASUs don't share one file.
        stem = e.get('structure_stem') or e['pdbcode']
        jobs.append({
            **e,
            'relaxed_pdb': str(relaxed_dir / f'{stem}.pdb'),
            'score_json': str(score_json),
            'log_path': str(logs_dir / f"{e['id']}.log"),
            'relax_repeats': args.relax_repeats,
            'timeout_s': args.timeout_s,
            'force_relax': args.force_relax,
            'python_exe': python_exe,
        })

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    print(f'Cached skip: {n_skip} | queued: {len(jobs)} | workers: {args.workers}')
    if jobs:
        print(f'  smallest queued: {jobs[0]["id"]} ({jobs[0]["nbytes"]} bytes)')
        print(f'  largest  queued: {jobs[-1]["id"]} ({jobs[-1]["nbytes"]} bytes)')
    if not jobs:
        summary = aggregate_energies(scores_dir, energies_out, key_mode=args.energy_key)
        print('Nothing to run. Current aggregate:')
        print(json.dumps(summary, indent=2))
        return

    n_ok = n_fail = n_crash = n_timeout = 0
    t_start = time.time()

    # ProcessPoolExecutor only schedules run_one (subprocess launcher); the heavy
    # PyRosetta work is isolated in child processes of those workers.
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_one, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            done += 1
            job = futures[fut]
            try:
                meta = fut.result()
            except Exception as e:
                meta = {
                    'entry_id': job['id'],
                    'pdbcode': job['pdbcode'],
                    'status': 'failed',
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'dG_separated': None,
                }

            append_jsonl(progress_path, meta)
            status = meta.get('status')
            if status == 'ok':
                n_ok += 1
                append_jsonl(successes_path, meta)
            else:
                n_fail += 1
                if status == 'crashed':
                    n_crash += 1
                if status == 'timeout':
                    n_timeout += 1
                append_jsonl(failures_path, meta)

            if done % 25 == 0 or done == len(jobs):
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1e-6)
                eta = (len(jobs) - done) / max(rate, 1e-6)
                print(
                    f'[{done}/{len(jobs)}] ok={n_ok} fail={n_fail} '
                    f'crash={n_crash} timeout={n_timeout} '
                    f'elapsed={elapsed/3600:.2f}h eta={eta/3600:.2f}h '
                    f'last={meta.get("entry_id")} status={status} '
                    f'dG={meta.get("dG_separated")}',
                    flush=True,
                )

            if done % 100 == 0:
                summary = aggregate_energies(scores_dir, energies_out, key_mode=args.energy_key)
                print(f'  checkpoint aggregate: {summary}', flush=True)

    summary = aggregate_energies(scores_dir, energies_out, key_mode=args.energy_key)
    print('\nFinished.')
    print(f'ok={n_ok} fail={n_fail} crash={n_crash} timeout={n_timeout}')
    print('Aggregate:', json.dumps(summary, indent=2))
    print(f'Failures log: {failures_path}')
    print(f'Per-job logs: {logs_dir}/')
    print(f'Energies: {energies_out}')
    print('\nTo copy into training cwd:')
    print(f'  cp {energies_out} precomputed_energies.pkl')
    print(f'  # and point dataset chothia_dir at {relaxed_dir}')


if __name__ == '__main__':
    # Avoid BLAS oversubscription inside each worker subprocess's parent launcher.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    main()
