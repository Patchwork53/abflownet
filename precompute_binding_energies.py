#!/usr/bin/env python
"""Precompute InterfaceAnalyzer binding energies (ΔG) for AbFlowNet TB rewards.

Paper: R(S^0) = exp(-α · BindingEnergy) with α = 1e-2 and BindingEnergy in
roughly [-100, 0] kcal/mol from PyRosetta InterfaceAnalyzer.

Example:
  python precompute_binding_energies.py \
      --chothia_dir data/all_structures/chothia \
      --summary_path data/sabdab_summary_all.tsv \
      --out precomputed_energies.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections import defaultdict

import pandas as pd
from tqdm.auto import tqdm


ALLOWED_AG_TYPES = {
    'protein',
    'protein | protein',
    'protein | protein | protein',
    'protein | protein | protein | protein | protein',
    'protein | protein | protein | protein',
}


def _nan_to_empty(val):
    if val != val or not val:
        return ''
    return val


def build_entries(summary_path):
    df = pd.read_csv(summary_path, sep='\t')
    entries = []
    for _, row in df.iterrows():
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
        Ag = _nan_to_empty(row.get('antigen_chain')).replace(' ', '')
        if not H and not L:
            continue
        pdb = str(row['pdb']).lower()
        entry_id = f'{pdb}_{H}_{L}_{Ag}'
        entries.append({
            'id': entry_id,
            'pdbcode': pdb,
            'H': H,
            'L': L,
            'Ag': Ag,
        })
    return entries


def interface_string(H, L, Ag):
    ab = ''.join([c for c in (H, L) if c and c != 'NA'])
    ag = ''.join([c for c in Ag.split('|') if c and c != 'NA'])
    if not ab or not ag:
        raise ValueError(f'Cannot form interface from H={H} L={L} Ag={Ag}')
    return f'{ab}_{ag}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chothia_dir', type=str, required=True)
    parser.add_argument('--summary_path', type=str, required=True)
    parser.add_argument('--out', type=str, default='precomputed_energies.pkl')
    parser.add_argument('--key', type=str, choices=['pdbcode', 'id'], default='pdbcode',
                        help='Dictionary key style. Train lookup accepts both.')
    args = parser.parse_args()

    # Local import so the rest of the repo can be used without PyRosetta.
    import pyrosetta
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', '2',
        '-no_fconfig',
    ]))

    def pyrosetta_interface_energy(pdb_path, interface):
        pose = pyrosetta.pose_from_pdb(pdb_path)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        mover.apply(pose)
        return pose.scores['dG_separated']

    entries = build_entries(args.summary_path)
    print(f'Entries after filtering: {len(entries)}')

    energies = {}
    errors = defaultdict(int)

    for entry in tqdm(entries, desc='InterfaceAnalyzer'):
        pdb_path = os.path.join(args.chothia_dir, f'{entry["pdbcode"]}.pdb')
        if not os.path.exists(pdb_path):
            errors['missing_pdb'] += 1
            continue
        try:
            iface = interface_string(entry['H'], entry['L'], entry['Ag'])
            dg = float(pyrosetta_interface_energy(pdb_path, iface))
        except Exception:
            errors['rosetta_fail'] += 1
            continue

        key = entry[args.key] if args.key == 'id' else entry['pdbcode']
        # Prefer the more negative (better) ΔG if a pdbcode appears multiple times.
        if key not in energies or dg < energies[key]:
            energies[key] = dg

    with open(args.out, 'wb') as f:
        pickle.dump(energies, f)

    vals = sorted(energies.values())
    if vals:
        median = vals[len(vals) // 2]
        frac = sum(1 for v in vals if -100 <= v <= 0) / len(vals)
        print(f'Wrote {len(energies)} energies to {args.out}')
        print(f'min={vals[0]:.2f} median={median:.2f} max={vals[-1]:.2f} frac_[-100,0]={frac:.3f}')
    else:
        print('Wrote 0 energies — check chothia_dir / PyRosetta.')
    print('Errors:', dict(errors))


if __name__ == '__main__':
    main()
