#!/usr/bin/env python
"""Compare InterfaceAnalyzer ΔG native vs PackRotamers vs FastRelax on one complex."""
from __future__ import annotations

import argparse
import re
import sys
import time


def log(msg):
    print(msg, flush=True)


def parse_interface(pdb_path, hchain, lchain, agchain):
    if hchain and lchain and lchain not in ('NA', 'nan'):
        ab = f'{hchain}{lchain}'
    elif hchain and hchain not in ('NA', 'nan'):
        ab = hchain
    else:
        ab = lchain
    ag = str(agchain).replace(' | ', '').replace('|', '').replace(' ', '')
    return f'{ab}_{ag}'


def ia_scores(pose, interface, sfxn):
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    out = {
        'dG_separated': float(pose.scores['dG_separated']),
        'E_total': float(sfxn(pose)),
    }
    for k in ('dG_separated/dSASAx100', 'sc_value', 'dSASA_int'):
        if k in pose.scores:
            out[k] = float(pose.scores[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdb', default='data/all_structures/chothia/5h8o.pdb')
    ap.add_argument('--H', default='A')
    ap.add_argument('--L', default='NA')
    ap.add_argument('--Ag', default='B')
    ap.add_argument('--relax_repeats', type=int, default=1)
    args = ap.parse_args()

    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
    from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

    log('Initializing PyRosetta...')
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', str(args.relax_repeats),
        '-no_fconfig',
    ]))

    interface = parse_interface(args.pdb, args.H, args.L, args.Ag)
    sfxn = pyrosetta.get_fa_scorefxn()
    log(f'PDB={args.pdb}')
    log(f'interface={interface}')

    log('Loading native pose...')
    t0 = time.time()
    pose_native = pyrosetta.pose_from_pdb(args.pdb)
    log(f'  residues={pose_native.total_residue()} load={time.time()-t0:.1f}s')

    log('Scoring native (InterfaceAnalyzer + pack_separated)...')
    t0 = time.time()
    native = ia_scores(pose_native, interface, sfxn)
    log(f'  {time.time()-t0:.1f}s -> {native}')

    log('PackRotamers (side chains only)...')
    pose_pack = pyrosetta.pose_from_pdb(args.pdb)
    tf = TaskFactory()
    tf.push_back(RestrictToRepacking())
    packer = PackRotamersMover(sfxn)
    packer.task_factory(tf)
    t0 = time.time()
    packer.apply(pose_pack)
    t_pack = time.time() - t0
    log(f'  pack done in {t_pack:.1f}s; scoring...')
    t0 = time.time()
    packed = ia_scores(pose_pack, interface, sfxn)
    log(f'  {time.time()-t0:.1f}s -> {packed}')

    log(f'FastRelax ({args.relax_repeats} repeat(s), constrained to start coords)...')
    pose_relax = pyrosetta.pose_from_pdb(args.pdb)
    relax = FastRelax(sfxn, args.relax_repeats)
    relax.constrain_relax_to_start_coords(True)
    t0 = time.time()
    relax.apply(pose_relax)
    t_relax = time.time() - t0
    log(f'  relax done in {t_relax:.1f}s; scoring...')
    t0 = time.time()
    relaxed = ia_scores(pose_relax, interface, sfxn)
    log(f'  {time.time()-t0:.1f}s -> {relaxed}')

    log('')
    log(f'{"metric":<28} {"native":>12} {"pack":>12} {"relax":>12} {"Δrelax":>12}')
    log('-' * 80)
    for k in native:
        b, p, r = native[k], packed[k], relaxed[k]
        log(f'{k:<28} {b:12.3f} {p:12.3f} {r:12.3f} {r-b:12.3f}')


if __name__ == '__main__':
    main()
