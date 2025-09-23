# AbFlowNet

This codebase was adapted from [[DiffAb]](https://github.com/luost26/diffab)

## Install

### Environment

```bash
conda env create -f env.yaml -n abflownet
conda activate abflownet
```

The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Datasets and Trained Weights

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder. 

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

Trained model weights are available [**here** (Hugging Face)](https://huggingface.co/luost26/DiffAb/tree/main) or [**here** (Google Drive)](https://drive.google.com/drive/folders/15ANqouWRTG2UmQS_p0ErSsrKsU4HmNQc?usp=sharing).


### [Optional] PyRosetta

PyRosetta is required to relax the generated structures and compute binding energy. Please follow the instruction [**here**](https://www.pyrosetta.org/downloads) to install.


## Train AbFlowNet (For RAbD Benchmarking)

```
python train.py configs/train/codesign_single.yml
```

Default training parameters `max_iters=200_000`, `start_tb_after=195_000` and `train.loss_weights.tb=0.000005` are set in `configs/train/codesign_single.yml` file. We precomputed the binding energies with PyRosetta and saved the results in `precomputed_energies.pkl`, used in `train.py`.

## Test On RAbD
The trained model weights are in `trained_models`

Run `./generate_test.sh` to generate CDRs on the 58 RAbD test complexes using `trained_models\abflownet.pt`

### For energy calculation
`python ./abflownet/tools/eval/run.py --pfx="" --root <path to generated pdbs>`

A `summary.csv` will be created in the same directory where generated pdbs are.

run `energy_eval.py` and give the path of the generated `summary.csv` as `--csv_path` as input argument.



## Switching to DiffAb test complexes
The code for loading the DiffAb test complexes are in `diffab/datasets/sabdab_diffab.py`


## Visualization of Proteins

For the visualization of antibody–antigen interactions, we used the 5MES protein complex. This structure features a broadly neutralizing antibody bound to the HIV-1 envelope glycoprotein (gp120), offering a clear view of the paratope-epitope interface. We used [**PyMOL**](https://pymol.org/). The following script can be used to generate a consistent view highlighting specific complementarity-determining regions (CDRs).

Save the following as a `.pml` script and run it in PyMOL after loading the desired PDB:

```pml
# Load the structure
load /path/to/your_structure.pdb

# Select antigen and antibody chains
select antigen, chain <antigen chain>
select antibody, chain <antibody chain>

# Hide default visuals
hide everything, all

# Show antigen as transparent surface
show surface, antigen
set transparency, 0.05, antigen
set surface_quality, 1

# Show antibody as cartoon
show cartoon, antibody
set cartoon_smooth_loops, on
set cartoon_flat_sheets, 1
set cartoon_transparency, 0.2, antibody

# Color everything else white
color white, antigen
color white, antibody

# Highlight region of interest (customize as needed)
select <region_of_interest>, chain <antibody chain> and resi <residue region>
show sticks, <region_of_interest>
color red, <region_of_interest>

# Enhance visualization settings
bg_color white
set ray_opaque_background, off
set two_sided_lighting, on
set ambient, 0.18
set specular, 0.3
set shininess, 50
set reflect, 0.2
set antialias, 2

# Focus view on the region of interest
orient <region_of_interest>
zoom <region_of_interest>, 10

# Export high-resolution image
ray 3000, 3000
png final_visualization.png, dpi=600
