import os
import sys
import yaml
import time
import pandas as pd
from pathlib import Path
from hashlib import sha512
from itertools import product
from datetime import datetime
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

    data_dir        = 'data'
    datasets        = ['CIFAR100']
    clients         = 50
    batch_size      = 32
    local_epochs    = 2
    global_rounds   = 60
    max_seed        = 20

    data_output_directory = Path(data_dir)
    data_output_directory.mkdir(parents=True, exist_ok=True)

    experiment_name, hyperparams = get_hyperparameters()
    areas = hyperparams['areas']

    a = 3

    if a == 0:
        algorithm = 'fedavg'
    elif a == 1:
        algorithm = 'fedproxy'
    elif a == 2:
        algorithm = 'scaffold'
    elif a == 3:
        algorithm = 'ifca'
    else:
        algorithm = 'Unknown'

    csv_file = f'finished_experiment_log.csv'

    df = pd.DataFrame(columns=['timestamp', 'experiment'])

    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        pass

    partitioning = 'Hard'
    #areas = [3, 5, 9]
    iid_start = time.time()
    for seed in range(max_seed):
        for dataset in datasets:
            for area in areas:
                simulator = Simulator(algorithm, partitioning, area, dataset, clients, batch_size, local_epochs, data_dir, seed)
                simulator.seed_everything(seed)
                simulator.start(global_rounds)
                experiment_name = f'seed-{seed}_regions-{area}_algorithm_{algorithm}'
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                new_line = {'timestamp': timestamp, 'experiment': experiment_name}
                df = pd.concat([df, pd.DataFrame([new_line])], ignore_index=True)
                df.to_csv(csv_file, index=False)