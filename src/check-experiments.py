import glob

def count_files_by_partitioning(partitioning):
    path = f'data/seed-*_algorithm-*_dataset-*_partitioning-{partitioning}_areas-*_clients-50.csv'
    files = list(glob.glob(path))
    print(f'Partitioning {partitioning} found {len(files)} files')


def count_files_by_seed_and_partitioning(seed, partitioning):
    path = f'data/seed-{seed}_algorithm-*_dataset-*_partitioning-{partitioning}_areas-*_clients-50.csv'
    files = list(glob.glob(path))
    print(f'Seed {seed} and partitioning {partitioning} found {len(files)} files')


if __name__ == "__main__":
    partitionings = ['iid', 'hard', 'dirichlet']
    seeds = range(5)

    for p in partitionings:
        count_files_by_partitioning(p)

    for seed in seeds:
        for partitioning in partitionings:
            count_files_by_seed_and_partitioning(seed, partitioning)