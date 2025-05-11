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


## Switching to DiffAb test complexes
The code for loading the DiffAb test complexes are in `diffab/datasets/sabdab_diffab.py`