import glob
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def load_data_from_csv(algorithm, dataset, partitioning, areas):
    files = glob.glob(f'data/seed-*_algorithm-{algorithm}_dataset-{dataset}_partitioning-{partitioning}_areas-{areas}_clients-50.csv')
    print(f'For algorithm {algorithm}, dataset {dataset}, partitioning {partitioning}, areas {areas} found {len(files)} files')
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)
    return dataframes

def compute_mean_variance(dfs):
    stacked = pd.concat(dfs, axis=0).groupby(level=0)
    mean_df = stacked.mean()
    variance_df = stacked.var(ddof=0)
    return mean_df, variance_df

def plot(data, partitioning, dataset, areas, metrics):
    colors = sns.color_palette("viridis", n_colors=len(data))
    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 6))
        for j, (algorithm, (mean_df, variance_df)) in enumerate(data.items()):
            sns.lineplot(
                data = mean_df,
                x = 'Round',
                y = metric,
                label = algorithm,
                color = colors[j],
            )
            mean = mean_df[metric]
            variance = variance_df[metric]
            upper_bound = mean + np.sqrt(variance)
            lower_bound = mean - np.sqrt(variance)
            plt.fill_between(mean.index, lower_bound, upper_bound, color=colors[j], alpha=0.2)
        plt.title(f'Dataset {dataset} - Partitioning {partitioning} - Areas {areas}')
        plt.xlabel('Round')
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'charts/{metric}_dataset-{dataset}_areas-{areas}_partitioning-{partitioning}.png', dpi=300)
        plt.close()

if __name__ == '__main__':
    matplotlib.rcParams.update({'axes.titlesize': 20})
    matplotlib.rcParams.update({'axes.labelsize': 18})
    matplotlib.rcParams.update({'xtick.labelsize': 15})
    matplotlib.rcParams.update({'ytick.labelsize': 15})
    plt.rcParams.update({"text.usetex": True})
    plt.rc('text.latex', preamble=r'\usepackage{amsmath,amssymb,amsfonts}')

    input_dir = 'data/'
    charts_dir = 'charts/'
    max_seed = 5
    metrics = ['TrainingLoss', 'ValidationLoss', 'ValidationAccuracy']

    Path(charts_dir).mkdir(parents=True, exist_ok=True)



    # IID charts
    # TODO

    # Non-IID charts
    partitionings = ['dirichlet'] # TODO - add hard
    datasets = ['MNIST', 'FashionMNIST', 'EMNIST']
    algorithms = ['fedavg', 'fedproxy', 'scaffold']
    areas = [3, 5, 9]
    for partitioning in partitionings:
        for dataset in datasets:
            for area in areas:
                data = {}
                print(f'Charting partitioning {partitioning} dataset {dataset} area {area}')
                for algorithm in algorithms:
                    dfs = load_data_from_csv(algorithm, dataset, partitioning, area)
                    mean, variance = compute_mean_variance(dfs)
                    data[algorithm] = (mean, variance)
                plot(data, partitioning, dataset, area, metrics)







