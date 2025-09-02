import math
import random
import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from models.MNIST import NNMnist
from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset, Subset, DataLoader

def initialize_model(name):
    if name == 'MNIST' or name == 'FashionMNIST':
        return NNMnist()
    elif name == 'EMNIST':
        return NNMnist(output_size=27)
    elif name == 'CIFAR100':
        model = mobilenet_v2(pretrained=True)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 100),
            nn.LogSoftmax(dim=1)
        )
        return model
    else:
        raise Exception(f'Model {name} not implemented! Please check :)')

def initialize_control_state(experiment, device):
    control_state = initialize_model(experiment).to(device)
    for param in control_state.parameters():
        nn.init.constant_(param, 0.0)
    return control_state.state_dict()

def hard_non_iid_mapping(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1 / len(elems)
    return distribution

def non_iid_mapping_validation(areas: int, labels: int) -> np.ndarray:
    labels_set = np.arange(labels)
    split_classes_per_area = np.array_split(labels_set, areas)
    distribution = np.zeros((areas, labels))
    for i, elems in enumerate(split_classes_per_area):
        rows = [i for _ in elems]
        distribution[rows, elems] = 1
    return distribution

def iid_mapping(areas: int, labels: int) -> np.ndarray:
    percentage = 1 / labels
    distribution = np.zeros((areas, labels))
    distribution.fill(percentage)
    return distribution

def partitioning(distribution: np.ndarray, data: Subset, test = False) -> dict[int, list[int]]:
    indices = list(range(len(data)))
    targets = data.targets if test else data.dataset.targets
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets)
    class_to_indices = find_class_to_indices(targets, indices)

    class_counts = torch.bincount(targets[indices])
    # class_to_indices = {}
    # for index in indices:
    #     c = targets[index].item()
    #     if c in class_to_indices:
    #         class_to_indices[c].append(index)
    #     else:
    #         class_to_indices[c] = [index]

    areas = distribution.shape[0]
    targets_cardinality = distribution.shape[1]
    max_examples_per_area = int(math.floor(len(indices) / areas))
    elements_per_class =  torch.floor(torch.tensor(distribution) * max_examples_per_area).to(torch.int)
    partitions = { a: [] for a in range(areas) }
    for area in range(areas):
        elements_per_class_in_area = elements_per_class[area, :].tolist()
        for c in sorted(class_to_indices.keys()):
            elements = min(elements_per_class_in_area[c], class_counts[c].item())
            selected_indices = random.sample(class_to_indices[c], elements)
            partitions[area].extend(selected_indices)
    return partitions

def find_class_to_indices(targets, indices) -> dict[int, list[int]]:
    class_to_indices = {}
    for index in indices:
        target = targets[index]
        if not isinstance(target, int):
            c = target.item()
        else: 
            c = target 
        if c in class_to_indices:
            class_to_indices[c].append(index)
        else:
            class_to_indices[c] = [index]
    return class_to_indices

def dirichlet_partitioning(data: Subset, areas: int, beta: float) -> dict[int, list[int]]:
    # Implemented as in: https://proceedings.mlr.press/v97/yurochkin19a.html
    min_size = 0
    indices = data.indices
    targets = data.dataset.targets
    N = len(indices)
    class_to_indices = {}
    for index in indices:
        c = targets[index].item()
        if c in class_to_indices:
            class_to_indices[c].append(index)
        else:
            class_to_indices[c] = [index]
    partitions = {a: [] for a in range(areas)}
    while min_size < 10:
        idx_batch = [[] for _ in range(areas)]
        for k in sorted(class_to_indices.keys()):
            idx_k = class_to_indices[k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, areas))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / areas) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(areas):
        np.random.shuffle(idx_batch[j])
        partitions[j] = idx_batch[j]
    return partitions

def test_model(model, dataset, batch_size, device):
    criterion = nn.NLLLoss()
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for batch_index, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)
    accuracy = correct / total
    return loss, accuracy

def plot_heatmap(data, labels, areas, name, floating = True):
    sns.heatmap(data, annot=True, cmap="YlGnBu",
                xticklabels=[f'{i}' for i in range(labels)],
                yticklabels=[f'{i}' for i in range(areas)],
                fmt= '.3f' if floating else 'd'
                )
    plt.xlabel('Label')
    plt.ylabel('Area')
    plt.tight_layout()
    plt.savefig(f'{name}.pdf')
    plt.close()