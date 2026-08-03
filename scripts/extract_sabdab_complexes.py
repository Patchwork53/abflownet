#!/usr/bin/env python
"""Extract one Ab–Ag complex per SAbDab entry into its own PDB.

SAbDab Chothia files often contain multiple crystal copies. This writes
``{entry_id}.pdb`` containing only the chains listed for that row
(Hchain, Lchain, antigen_chain), so FastRelax/InterfaceAnalyzer can run
on a single complex without changing the relax protocol.

Example:
  python scripts/extract_sabdab_complexes.py \\
    --chothia_dir data/all_structures/chothia \\
    --summary_path data/sabdab_summary_all.tsv \\
    --out_dir data/chothia_complexes
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}


def _nan_to_empty(val) -> str:
    if val is None:
        return ''
    if isinstance(val, float) and val != val:
        return ''
    s = str(val).strip()
    if s.lower() in ('', 'nan', 'none', 'na'):
        return ''
    return s


def _keep_chains(H: str, L: str, Ag_raw: str) -> Set[str]:
    keep = set()
    if H:
        keep.add(H)
    if L:
        keep.add(L)
    for c in Ag_raw.replace(' ', '').split('|'):
        if c and c != 'NA':
            keep.add(c)
    return keep


def build_entries(summary_path: str, chothia_dir: str) -> List[dict]:
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
            Ag_raw = _nan_to_empty(row.get('antigen_chain')).replace(' ', '')
            if not H and not L:
                continue
            if not Ag_raw:
                continue

            pdb = str(row['pdb']).lower()
            in_pdb = os.path.join(chothia_dir, f'{pdb}.pdb')
            if not os.path.isfile(in_pdb):
                continue

            Ag_id = Ag_raw.replace('|', '')
            entry_id = f'{pdb}_{H}_{L}_{Ag_id}'
            keep = _keep_chains(H, L, Ag_raw)
            if not keep:
                continue
            entries.append({
                'id': entry_id,
                'pdbcode': pdb,
                'H': H,
                'L': L,
                'Ag': Ag_raw,
                'in_pdb': in_pdb,
                'keep': ''.join(sorted(keep)),  # for ProcessPool pickling as str
            })
    return entries


def extract_one(job: dict) -> dict:
    """Filter ATOM/HETATM/TER lines to keep-chains; drop waters."""
    keep = set(job['keep'])
    out_pdb = job['out_pdb']
    in_pdb = job['in_pdb']
    n_in = n_out = 0
    chains_seen = set()
    try:
        os.makedirs(os.path.dirname(out_pdb) or '.', exist_ok=True)
        tmp = out_pdb + '.tmp'
        with open(in_pdb) as fin, open(tmp, 'w') as fout:
            for line in fin:
                if line.startswith(('ATOM', 'HETATM')):
                    n_in += 1
                    if len(line) < 22:
                        continue
                    ch = line[21]
                    resname = line[17:20].strip()
                    if ch not in keep:
                        continue
                    if resname in ('HOH', 'WAT', 'DOD'):
                        continue
                    chains_seen.add(ch)
                    fout.write(line)
                    n_out += 1
                elif line.startswith('TER'):
                    if len(line) > 21 and line[21] in keep:
                        fout.write(line)
                elif line.startswith('END'):
                    fout.write('END\n')
            if n_out == 0:
                raise RuntimeError(f'no atoms kept for chains {sorted(keep)}')
            missing = keep - chains_seen
            if missing:
                raise RuntimeError(
                    f'missing chains {sorted(missing)} in {in_pdb} '
                    f'(found {sorted(chains_seen)})'
                )
        os.replace(tmp, out_pdb)
        return {
            'id': job['id'],
            'status': 'ok',
            'n_atoms_in': n_in,
            'n_atoms_out': n_out,
            'chains': ''.join(sorted(chains_seen)),
            'out_pdb': out_pdb,
            'error': None,
        }
    except Exception as e:
        try:
            if os.path.isfile(out_pdb + '.tmp'):
                os.remove(out_pdb + '.tmp')
        except OSError:
            pass
        return {
            'id': job['id'],
            'status': 'failed',
            'n_atoms_in': n_in,
            'n_atoms_out': n_out,
            'chains': '',
            'out_pdb': out_pdb,
            'error': str(e),
        }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chothia_dir', default='data/all_structures/chothia')
    ap.add_argument('--summary_path', default='data/sabdab_summary_all.tsv')
    ap.add_argument('--out_dir', default='data/chothia_complexes')
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = build_entries(args.summary_path, args.chothia_dir)
    print(f'Entries to extract: {len(entries)}')

    jobs = []
    n_skip = 0
    for e in entries:
        out_pdb = str(out_dir / f"{e['id']}.pdb")
        if (not args.force) and os.path.isfile(out_pdb) and os.path.getsize(out_pdb) > 0:
            n_skip += 1
            continue
        jobs.append({**e, 'out_pdb': out_pdb})

    if args.limit and args.limit > 0:
        jobs = jobs[: args.limit]

    print(f'Skip existing: {n_skip} | queued: {len(jobs)} | workers: {args.workers}')

    n_ok = n_fail = 0
    failures = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(extract_one, j): j for j in jobs}
            done = 0
            for fut in as_completed(futs):
                done += 1
                meta = fut.result()
                if meta['status'] == 'ok':
                    n_ok += 1
                else:
                    n_fail += 1
                    failures.append(meta)
                if done % 500 == 0 or done == len(jobs):
                    print(f'[{done}/{len(jobs)}] ok={n_ok} fail={n_fail} last={meta["id"]} status={meta["status"]}', flush=True)

    fail_path = out_dir / 'extract_failures.jsonl'
    if failures:
        with open(fail_path, 'w') as f:
            import json
            for row in failures:
                f.write(json.dumps(row) + '\n')
        print(f'Wrote {len(failures)} failures -> {fail_path}')

    n_files = sum(1 for _ in out_dir.glob('*.pdb'))
    print(f'Done. ok={n_ok} fail={n_fail} | PDBs on disk: {n_files}')
    print(f'Next: point FastRelax at this directory (entry_id filenames):')
    print(
        f'  python scripts/launch_fastrelax.py \\\n'
        f'    --chothia_dir {out_dir} \\\n'
        f'    --summary_path {args.summary_path} \\\n'
        f'    --out_dir data/fastrelax_cache \\\n'
        f'    --structure_name entry_id \\\n'
        f'    --workers 32 --force --force_relax'
    )
    return 0 if n_fail == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
