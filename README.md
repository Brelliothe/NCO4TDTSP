# [NeurIPS 2025] Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem
Official implementation of the paper: [Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem](https://openreview.net/pdf?id=UXTR6ZYV1x).
## About
This repository contains the code and data to reproduce the experiments for our NeurIPS 2025 paper. We propose a Neural Combinatorial Optimization (NCO) approach to solve the Time-Dependent Traveling Salesman Problem (TDTSP), with validated ability to learn the spatiotemporal dynamics.
## Quick Start
Follow these steps to set up the environment and reproduce the experiments.

1. Installation

Clone the repository and install the dependencies in editable mode:
```Bash
git clone git@github.com:Brelliothe/NCO4TDTSP.git
cd NCO4TDTSP
pip install -e .
```
2. Data Preparation

We provide processed datasets for Beijing and Lyon directly in this repository.
* Full processed dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1JsWs0MwUXAyXLbD6VXeW4_GR8Utcuqz-/view?usp=drive_link). 
* Customized dataset should be formed as a numpy array in shape `[nodes, nodes, horizon]`. 
* All processed data files should be put `/data/tdtsp` folder by default. 

3. Training 
To train the model using the default configuration (Beijing dataset, size 10):
```Bash
python run.py
```
To change the dataset, problem size, or hyperparameters, modify the config file located at `configs/experiment/routing/tdtsp-matnet.yaml`.

4. Evaluation. 
To reproduce the experimental results:
* Substitute the model path in `comp.py` and `experiments.py`, 
* Run the evaluation scripts:
```Bash
# Compare with baselines on full datasets
python comp.py
# Evaluate on selected datasets
python experiments.py
```
to reproduce the experiments.

## Repository Structure
The code is built upon repo [RL4CO](https://github.com/ai4co/rl4co). The architecture is as follows:
```Plaintext
├── run.py                                  # Entry point for training
├── data/
│   └── tdtsp/                              # Time-dependent adjacency matrices
├── configs/
│   └── experiment/routing/tdtsp-matnet.yaml # Training configuration
├── testcases/
│   ├── location_20_dataset_10000.pt        # 10k TDTSP instances (20 nodes)
│   └── location_20_dataset_10000/          # Solutions (DP) for TDTSP and ATSP
├── rl4co/
│   ├── baselines/                          # Baseline algorithms for comparison
│   ├── envs/routing/tdtsp/                 # TDTSP RL environment
│   ├── models/zoo/tmatnet/                 # Model architecture implementations
│   └── tasks/
│       ├── collect_numba_nobatch.py        # Numba solver for ground truth generation
│       ├── comp.py                         # Baseline comparison script
│       ├── data.py                         # Data cleaning and processing
│       └── experiments.py                  # Evaluation script
└── README.md
```

## Data References

The processed data is provided [here](https://drive.google.com/file/d/1JsWs0MwUXAyXLbD6VXeW4_GR8Utcuqz-/view?usp=drive_link). Raw data sources are listed below:

* [1] [Beijing data](https://github.com/hachinoone/DRLSolver4DTSP/blob/main/data_nodes/node_19.txt): Zhang et al. (2021) [Solving Dynamic Traveling Salesman Problems With Deep Reinforcement Learning](https://ieeexplore.ieee.org/document/9537638).
* [2] [Lyon data](https://perso.citi-lab.fr/csolnon/TDTSP.html): Melgarejo et al. (2015) [A Time-Dependent No-Overlap Constraint: Application to Urban Delivery Problems](https://link.springer.com/content/pdf/10.1007/978-3-319-18008-3_1).

* [3] [10 cities data](https://gitlab.com/muelleratorunibonnde/vrptdt-benchmark), Blauth et al. (2022) [Vehicle Routing with Time-Dependent Travel Times: Theory, Practice, and Benchmarks](https://arxiv.org/abs/2205.00889v2).

## Citation
If you find this work useful for your research, please consider citing our paper:
```
@inproceedings{yang2025neural,
  title={Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem},
  author={Ruixiao Yang and Chuchu Fan},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://openreview.net/pdf?id=UXTR6ZYV1x}
}
```

## Questions/Bugs

Please submit a Github issue or contact ruixiao@mit.edu if you have any questions or find any bugs.