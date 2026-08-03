# pyright: reportMissingImports=false
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

def compute_total_energy(pdb_path):
    """Rosetta total score (fa_scorefxn). Not the TB training reward."""
    pose = pyrosetta.pose_from_pdb(pdb_path)
    score_function = pyrosetta.get_fa_scorefxn()
    energy = score_function(pose) + 1e-8
    return energy


def pyrosetta_interface_energy(pdb_path, interface):
    """InterfaceAnalyzer ΔG (dG_separated). This is the AbFlowNet TB BindingEnergy."""
    pose = pyrosetta.pose_from_pdb(pdb_path)
    mover = InterfaceAnalyzerMover(interface)
    mover.set_pack_separated(True)
    mover.apply(pose)
    return pose.scores['dG_separated']


def eval_interface_energy(task):
    model_gen = task.get_gen_biopython_model()
    antigen_chains = set()
    for chain in model_gen:
        if chain.id not in task.ab_chains:
            antigen_chains.add(chain.id)
    antigen_chains = ''.join(list(antigen_chains))
    antibody_chains = ''.join(task.ab_chains)
    interface = f"{antibody_chains}_{antigen_chains}"

    dG = pyrosetta_interface_energy(task.in_path, interface)
    E_total = compute_total_energy(task.in_path)

    # Keep legacy keys used by energy_eval.py:
    #   dG_gen  -> InterfaceAnalyzer ΔG
    #   dG_ref  -> total Rosetta energy (misnamed historically; is E_total)
    task.scores.update({
        'dG_gen': dG,
        'dG_ref': E_total,
        'E_total': E_total,
        'ddG': dG - E_total,
    })
    return task
