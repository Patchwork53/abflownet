#!/usr/bin/env python
"""Single-complex FastRelax + InterfaceAnalyzer worker (subprocess-isolated).

Exits 0 on success, 1 on handled failure, 2 on unexpected error.
Never raises into the parent — parent reads JSON on stdout / status file.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp, path)


def _interface_string(H: str, L: str, Ag: str) -> str:
    """Rosetta InterfaceAnalyzer interface string, e.g. 'HL_A' or 'HL_AC'."""
    ab = ''.join(c for c in (H, L) if c and c != 'NA')
    # Multi-chain antigens in SAbDab are 'A | C'; IA wants concatenated chain ids.
    ag = ''.join(c for c in Ag.replace(' ', '').split('|') if c and c != 'NA')
    if not ab or not ag:
        raise ValueError(f'bad interface H={H!r} L={L!r} Ag={Ag!r}')
    return f'{ab}_{ag}'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--entry_id', required=True)
    ap.add_argument('--pdbcode', required=True)
    ap.add_argument('--H', default='')
    ap.add_argument('--L', default='')
    ap.add_argument('--Ag', required=True)
    ap.add_argument('--in_pdb', required=True)
    ap.add_argument('--relaxed_pdb', required=True)
    ap.add_argument('--score_json', required=True)
    ap.add_argument('--relax_repeats', type=int, default=1)
    ap.add_argument('--skip_relax_if_cached', action='store_true', default=True)
    ap.add_argument('--force_relax', action='store_true')
    args = ap.parse_args()

    t_all = time.time()
    result = {
        'entry_id': args.entry_id,
        'pdbcode': args.pdbcode,
        'status': 'failed',
        'error': None,
        'error_type': None,
        'interface': None,
        'dG_separated': None,
        'E_total': None,
        'nres': None,
        't_relax_s': None,
        't_score_s': None,
        't_total_s': None,
        'relaxed_pdb': args.relaxed_pdb,
        'used_cached_relax': False,
    }

    try:
        if not os.path.isfile(args.in_pdb):
            raise FileNotFoundError(f'missing input pdb: {args.in_pdb}')

        interface = _interface_string(args.H, args.L, args.Ag)
        result['interface'] = interface

        import pyrosetta
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

        pyrosetta.init(' '.join([
            '-mute', 'all',
            '-use_input_sc',
            '-ignore_unrecognized_res',
            '-ignore_zero_occupancy', 'false',
            '-load_PDB_components', 'false',
            '-relax:default_repeats', str(args.relax_repeats),
            '-no_fconfig',
        ]))
        sfxn = pyrosetta.get_fa_scorefxn()

        # --- FastRelax (cached per pdbcode) ---
        need_relax = args.force_relax or (not os.path.isfile(args.relaxed_pdb))
        if need_relax:
            os.makedirs(os.path.dirname(args.relaxed_pdb) or '.', exist_ok=True)
            lock_path = args.relaxed_pdb + '.lock'
            # Exclusive lock so concurrent workers sharing a pdbcode don't double-relax.
            import fcntl
            with open(lock_path, 'w') as lockf:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
                if (not args.force_relax) and os.path.isfile(args.relaxed_pdb):
                    result['used_cached_relax'] = True
                else:
                    pose = pyrosetta.pose_from_pdb(args.in_pdb)
                    result['nres'] = int(pose.total_residue())
                    relax = FastRelax(sfxn, args.relax_repeats)
                    relax.constrain_relax_to_start_coords(True)
                    t0 = time.time()
                    relax.apply(pose)
                    result['t_relax_s'] = time.time() - t0
                    # Dump PDB
                    tmp_pdb = args.relaxed_pdb + '.tmp'
                    pose.dump_pdb(tmp_pdb)
                    os.replace(tmp_pdb, args.relaxed_pdb)
        else:
            result['used_cached_relax'] = True

        if not os.path.isfile(args.relaxed_pdb):
            raise RuntimeError('relaxed pdb missing after relax step')

        # --- InterfaceAnalyzer on relaxed structure ---
        pose_r = pyrosetta.pose_from_pdb(args.relaxed_pdb)
        if result['nres'] is None:
            result['nres'] = int(pose_r.total_residue())
        t0 = time.time()
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        mover.apply(pose_r)
        result['t_score_s'] = time.time() - t0
        result['dG_separated'] = float(pose_r.scores['dG_separated'])
        result['E_total'] = float(sfxn(pose_r))
        result['status'] = 'ok'
        result['t_total_s'] = time.time() - t_all
        _write_json(args.score_json, result)
        print(json.dumps(result), flush=True)
        return 0

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        result['error_type'] = type(e).__name__
        result['traceback'] = traceback.format_exc()
        result['t_total_s'] = time.time() - t_all
        try:
            _write_json(args.score_json, result)
        except Exception:
            pass
        print(json.dumps(result), flush=True)
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as e:  # noqa: BLE001 — last-resort crash barrier
        # Segfaults won't hit this; other fatal errors might.
        payload = {
            'status': 'crashed',
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': traceback.format_exc(),
        }
        print(json.dumps(payload), flush=True)
        sys.exit(2)
