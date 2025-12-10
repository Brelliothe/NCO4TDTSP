# Code for "Neural Combinatorial Optimization for Time Dependent Traveling Salesman Problem"
This is the raw code for the project, we will provide a more organized version after published.

The code is based on project [RL4CO](https://github.com/ai4co/rl4co). The architecture is as follows:
```
├── README.md
├── run.py  # starter file
├── data/tdtsp  # data files, all the time-dependent adjacent matrices
├── configs  # configs for training our model
    ├── experiment/routing/tdtsp-matnet.yaml  # config for training the model
├── testcases  # test datasets for 20 nodes instances
    ├── location_20_dataset_10000.pt  # 10000 TDTSP instances  
    ├── location_20_dataset_10000  # TDTSP and ATSP solution given by DP 
├── rl4co  # code folder
    ├── baselines  # baselines for comparison
    ├── envs/routing/tdtsp  # environment for TDTSP
    ├── models/zoo/tmatnet   # model architectures
    ├── tasks
        ├── collect_numba_nobatch.py  # numba version of the TDTSP solver, solve the TDTSP instances for ground truths
        ├── comp.py  # compare our model with baselines on full datasets
        ├── data.py  # clean the data
        ├── experiments.py  # evaluate our model on selected datasets
```
Due to space limit, we only provide the dataset of Beijing and Lyon in the appendix material. 

To install, you can use the following command:
```
pip install -e .
```
To train the model, you can use the following command:
```
python run.py
```
To change the dataset or the problem size, you can modify the config file in `configs/experiment/routing/tdtsp-matnet.yaml`. The default dataset is `beijing` and the default problem size is 10.

To evaluate the model, you can substitute the model path in `comp.py` and run the following command:
```
python comp.py
```
and substitute the model path in `experiments.py` and run the following command:
```
python experiments.py
```