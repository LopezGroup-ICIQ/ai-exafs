<div align="center">

<h1 align="center"><img src="img/logo.png" width="250"/></h1>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
[![ChemRxiv](https://img.shields.io/badge/ChemRxiv-Preprint-red)](https://doi.org/10.26434/chemrxiv-2026-sv3f6)

</div>

## 📌 Introduction

AI-EXAFS is a Graph Neural Network (GNN) framework developed to predict X-ray absorption spectroscopy (XAS) signals, with a particular emphasis on the extended X-ray absorption fine structures (EXAFS) region. Trained on 86k transition metal complexes from the TmQM_wB97MV dataset[1,2] and compared against experimental data of single-atom catalysts, AI-EXAFS learns the complex structural dependencies that shape photoelectron scattering and interference effects. In this framework, graph edges are interpreted as scattering paths rather than conventional chemical bonds, enabling the model to encode the local coordination environment in a physically meaningful way. By bridging atomic structure and EXAFS response through learned representations, AI-EXAFS provides a pathway toward automated interpretation of experimental spectra and rational design of catalytic materials.

## ⚡ Installation
After installing [conda](http://conda.pydata.org/), run the following commands to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html)
named `ai-exafs` and install the dependencies.

```bash
conda env create -f env.yaml
conda activate ai-exafs
pre-commit install
```

## 🚀 Quickstart

### Generate LMDB file
To generate the `LMDB` file, first you need to place a pickle file in the `data` folder with all the raw information inside. We provide both the pickle file (see supplementary data) and the generation script in the `scripts` folder, so the user can either use our pickle file to re-generate the lmdb file or create a new pickle file from the original TmQM_wB97MV dataset[2].

Then, after getting the pickle file, the user needs to generate the `LMDB` file using the `create_lmdb.py` script as follows:

```python
from ai_exafs.preprocessing.graph import EXAFSGraph

# Paths
data_path = "./data/wf_tmQM_wB97MV.pkl"
db_path = "./data/wf_tmQM_wB97MV.lmdb"

# LMDB Generator
extractor = EXAFSGraph(
    data_path=data_path,  # Path where the pickle file is placed
    db_path=db_path,      # Path where the LMDB file will be located
    kmax=12.0,            # Maximum K to be trained on
    max_z=92,             # Maximum Z value of chemical elements
    spectra_dim=928,      # To interpolate the EXAFS spectra to a given dimension
)

# Run
extractor.get_lmdb()
# extractor.get_pickle()
```

This method automatically generates the molecular graph representations and stores the processed EXAFS spectra (K-space) inside the [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) `Data` object with other metadata. Finally, the LMDB file is exported in the same `data` folder.

### Training AI-EXAFS
To train AI-EXAFS, we use [hydra](https://hydra.cc/) to configure the model's hyperparameters and training settings through the `config.yaml` file stored in the `configs` folder. We provide the general configuration file `config.yaml` for training the model. We refer the user to check all the hyperparameters used in our work within the yaml files.

> #### NOTE:
> Please, change the experiment id inside the `config.yaml` file with a lable that tracks your experiments as (e.g., `id: "ai_exafs_experiment_1"`).
> Also, inside the `paths` the `root_dir` tag should be changed to the path where `ai-exafs` is cloned (e.g., `root_dir: "/home/USER/ai-exafs"`).

To train AI-EXAFS on the Graph-EXAFS relationship, use the following command:

```bash
python main.py --config-name config.yaml
```

The model's checkpoint files and the normalization parameters (as json file) are stored automatically in the `checkpoints` folder. The `ExafsFTOut` callbacks will save a json file in the `outputs` folder for further analysis of the model's performance in the test dataset. These outputs can be visualized with the provided `jupyter notebooks` (see Evaluate AI-EXAFS performance section).

We strongly recommend to use [wandb](https://docs.wandb.ai/) platform to track the training/validation/testing loss functions during the execution.

> #### NOTE:
> When training starts, a `processed` folder is created inside the `data` directory; if another training is launched, the system will automatically reuse the processed data, and to force reprocessing, simply delete the `processed` folder.

### Hyperparameter Optimization
For `HPO` we use [optuna](https://optuna.readthedocs.io/en/stable/) python library to optimize both the model's architecture and the training hyperparameters. We recommend the user to look at the `main_optuna.py` script before launching an `HPO` experiment.

### Evaluate AI-EXAFS performance
We provide three jupyter notebooks, along with the pickle files (see supplementary data section) with all the testing data, in the folder `notebooks` to analyze and visualize the performance of AI-EXAFS to get the same plots that are in the publication manuscript.

- `notebooks/01_ai_exafs_performance.ipynb`: To plot the error histogram divided in quartiles and some examples from the testing dataset (10%).
- `notebooks/02_ablation_study.ipynb`: To plot the ablation study based on the different parts of the model and architecture design.
- `notebooks/03_experimental_comparison.ipynb`: To plot the comparison between K2PtCl6 complex and the supported Pt SAC catalyst.

The `notebooks/outputs` folder is to place the callback pickle files for analysis.

### Makefile
Inside the `Makefile` there are a few handy commands to streamline cleaning tasks.

```bash
make wandb-sync  # In case of using wandb offline

make clean-data  # To remove the processed folder from PyG

make clean-all   # To clean __pycache__ folders and unnecessary things
```

## 💾 Supplementary Data
The supplemetary data has been published in a [Zenodo]() repository, providing the datasets in both pickle and LMDB formats, the trained `AI-EXAFS` checkpoints for all experiments, and the output callback pickle files to ensure full repoducibility of the reported results.

> #### NOTE:
> To reproduce the reported results, the dataset pickle and LMDB files should be placed in the `data` folder, the pre-trained checkpoints in the `pre_trained` folder, and the outputs from the callback can be visualize by placing them within the `notebooks/outputs` folder.

## ❤️  Acknowledgements
This publication was created as part of NCCR Catalysis (grant numbers 180544 and 225147), a National Centre of Competence in Research funded by the Swiss National Science Foundation. The authors thank BSC-RES for generously providing computational resources. The authors also acknowledge the ANIMATE project (PID2024-157556OB-I00), funded by MCIU/AEI/10.13039/501100011033/FEDER, UE. The authors are also grateful to Dr. Olga Safonova from the Paul Scherrer Institute (PSI) for fruitful discussions and valuable feedback throughout the project.

## ⚖️  License
`AI-EXAFS` is released under the [MIT](https://github.com/LopezGroup-ICIQ/ai-exafs/blob/main/LICENSE.md) license.

## 📄 Citing `AI-EXAFS`

If you use this codebase in your work, please consider citing:

```bibtex
@article{aiexafs,
  title={Learning EXAFS from atomic structure through physics-informed machine learning},
  author={Heras-Domingo, Javier and Ruiz-Ferrando, Andrea and H. Clark, Adam and Pollitt, Stephan and Mitchell,
          Sharon and V. Safonova, Olga and P{\'e}rez-Ram{\'\i}rez, Javier and L{\'o}pez, N{\'u}ria},
  year={2026},
  publisher={ChemRxiv},
  doi={https://doi.org/10.26434/chemrxiv-2026-sv3f6}
}
```

## References
<a id="1">[1]</a>
Balcells, D., Skjelstad, B. B., tmQM dataset - Quantum geometries and properties of 86k transition metal complexes. Journal of Chemical Information and Modeling (JCIM), 2020, 60(12), 6135-6146. [URL](https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c01041)

<a id="2">[2]</a>
Garrison, A. G., Heras-Domingo, J., Kitchin, J. R., dos Passos Gomes, G., Ulissi, Z. W., Blau, S. M., Applying Large Graph Neural Networks to Predict Transition Metal Complex Energies sing the tmQM_wB97MV Data set. Journal of Chemical Information and Modeling, 2023, 63(24), 7642-7654. [URL](https://pubs.acs.org/doi/full/10.1021/acs.jcim.3c01226).
