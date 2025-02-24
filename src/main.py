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
    hyperparams = os.environ[HYPERPARAMETERS_NAME]
    print(hyperparams)
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

    total_experiments = 0

    # datasets        = ['MNIST', 'FashionMNIST', 'EMNIST']
    datasets        = ['EMNIST']
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 30
    data_dir        = 'data-iot'
    max_seed        = 20

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    # # Experiments IID
    # partitioning = 'iid'
    # experiment_name = 'fedavg'
    # areas = 3
    # iid_start = time.time()
    # for seed in range(max_seed):
    #     seed_start = time.time()
    #     for dataset in datasets:
    #         simulator = Simulator(experiment_name, partitioning, areas, dataset, clients, batch_size, local_epochs, data_dir, seed)
    #         simulator.seed_everything(seed)
    #         simulator.start(global_rounds)
    #         total_experiments += 1
    #     seed_end = time.time()
    #     print(f'Seed {seed} took {seed_end - seed_start} seconds')
    # iid_end = time.time()
    # print(f'IID experiments took {iid_end - iid_start} seconds')

    # Experiments non-IID dirichlet
    # partitioning = 'dirichlet'
    # experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    # # experiment_names = ['fedavg']
    # areas = [3, 6, 9]
    # non_iid_start = time.time()
    # for seed in range(max_seed):
    #     seed_start = time.time()
    #     for experiment_name in experiment_names:
    #         for dataset in datasets:
    #             for area in areas:
    #                 print(f'starting dirichlet seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
    #                 simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
    #                 simulator.seed_everything(seed)
    #                 simulator.start(global_rounds)
    #                 total_experiments += 1
    #     seed_end = time.time()
    #     print(f'Seed {seed} took {seed_end - seed_start} seconds')
    # non_iid_end = time.time()
    # print(f'non-IID experiments took {non_iid_end - non_iid_start} seconds')

    # Experiments non-IID hard EMNIST
    partitioning = 'hard'
    # experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    experiment_names = ['fedproxy', 'scaffold']
    areas = [3, 5, 9]
    non_iid_start = time.time()
    for seed in range(max_seed):
        seed_start = time.time()
        for experiment_name in experiment_names:
            for dataset in ['EMNIST']:
                for area in areas:
                    print(f'starting hard seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
                    simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                    simulator.seed_everything(seed)
                    simulator.start(global_rounds)
                    total_experiments += 1
        seed_end = time.time()
        print(f'Seed {seed} took {seed_end - seed_start} seconds')
    non_iid_end = time.time()
    print(f'non-IID experiments hard EMNIST took {non_iid_end - non_iid_start} seconds')
    #
    # # Experiments non-IID hard MNIST and Fashion
    # partitioning = 'hard'
    # experiment_names = ['fedavg', 'fedproxy', 'scaffold']
    # areas = [3]
    # non_iid_start = time.time()
    # for seed in range(max_seed):
    #     seed_start = time.time()
    #     for experiment_name in experiment_names:
    #         for dataset in ['MNIST', 'FashionMNIST']:
    #             for area in areas:
    #                 print(f'starting hard seed {seed} experiment {experiment_name} dataset {dataset} area {area}')
    #                 simulator = Simulator(experiment_name, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
    #                 simulator.seed_everything(seed)
    #                 simulator.start(global_rounds)
    #     seed_end = time.time()
    #     print(f'Seed {seed} took {seed_end - seed_start} seconds')
    # non_iid_end = time.time()
    # print(f'non-IID experiments hard MNIST and Fashion took {non_iid_end - non_iid_start} seconds')
    # #
    # #
    # print(f'Total experiments {total_experiments}')