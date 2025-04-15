import os
import sys
import yaml
import time
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from simulator.Simulator import Simulator

def get_hyperparameters():
    """
    Fetches the hyperparameters from the docker compose config file
    :return: the experiment name and the hyperparameters (as a dictionary name -> values)
    """
    hyperparams = os.environ['LEARNING_HYPERPARAMETERS']
    hyperparams = yaml.safe_load(hyperparams)
    experiment_name, hyperparams = list(hyperparams.items())[0]
    return experiment_name.lower(), hyperparams


def check_hyperparameters(hyperparams):
    """
    Checks if the hyperparameters fetched from the config file are valid
    :param hyperparams: all the hyperparameters fetched from the configuration file
    :raise: raises a ValueError if the hyperparameters are invalid
    """
    valid_hyperparams = ['partitioning', 'areas', 'seed', 'dataset', 'clients']
    for index, hp in enumerate(hyperparams.keys()):
        if hp != valid_hyperparams[index]:
            raise ValueError(f'''
                The hyperparameter {hp} is not valid! 
                Valid hyperparameters are: {valid_hyperparams} (they must be in this exact order)
            ''')


if __name__ == '__main__':

    data_dir        = Path(os.getenv('DATA_DIR', './data'))
    datasets        = ['EMNIST']
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 30
    max_seed        = 20

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    experiment_name, hyperparams = get_hyperparameters()
    #
    # experiment_name = 'ifca'
    # experiment_name = 'fedavg'
    # hyperparams = {'areas' : [3, 5, 9]}

    clusters = {
        3 : [1, 2], 
        5 : [1, 2, 3],
        9 : [1, 3, 5, 7]
    }

    # Experiments non-IID hard EMNIST
    partitioning = 'hard'
    areas = hyperparams['areas']
    for seed in range(max_seed):
        for dataset in ['EMNIST']:
            for area in areas:
            # for area in [9]:
                for cls in clusters[area]: 
                    print(f'starting hard seed {seed} experiment {experiment_name} dataset {dataset} area {area} clusters: {cls}')
                    simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed, number_of_clusters=cls)
                    simulator.seed_everything(seed)
                    simulator.start(global_rounds)
