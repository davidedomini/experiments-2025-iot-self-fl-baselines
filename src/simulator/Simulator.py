import torch
import random
import numpy as np
import pandas as pd
import utils.FedUtils as utils
from collections import Counter
from torchvision import datasets, transforms
from client.FedAvgClient import FedAvgClient
from client.FedProxyClient import FedProxyClient
from client.IFCAClient import IFCAClient
from server.FedAvgServer import FedAvgServer
from client.ScaffoldClient import ScaffoldClient
from server.ScaffoldServer import ScaffoldServer
from server.IFCAServer import IFCAServer
from torch.utils.data import Subset, random_split

class Simulator:

    def __init__(self, algorithm, partitioning, areas, dataset_name, n_clients, batch_size, local_epochs, data_folder, seed, number_of_clusters = 0):
        self.mapping_client_data_validation = {}
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.dataset_name = dataset_name
        self.number_of_clusters = number_of_clusters
        self.complete_dataset, self.training_data, self.validation_data = self.initialize_data()
        self.partitioning = partitioning
        self.algorithm = algorithm
        self.areas = areas
        self.n_clients = n_clients
        if self.algorithm == 'ifca':
            self.export_path = f'{data_folder}/seed-{seed}_algorithm-{self.algorithm}_dataset-{dataset_name}_partitioning-{self.partitioning}_areas-{self.areas}_clusters-{self.number_of_clusters}_clients-{self.n_clients}'
        else:
            self.export_path = f'{data_folder}/seed-{seed}_algorithm-{self.algorithm}_dataset-{dataset_name}_partitioning-{self.partitioning}_areas-{self.areas}_clients-{self.n_clients}'
        self.simulation_data = pd.DataFrame(columns=['Round','TrainingLoss', 'ValidationLoss', 'ValidationAccuracy'])
        self.clients = self.initialize_clients()
        self.server = self.initialize_server()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seed_everything(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def start(self, global_rounds):
        for r in range(global_rounds):
            print(f'Starting global round {r}')
            self.notify_clients()
            training_loss = self.clients_update()
            self.notify_server()
            self.server_update()
            validation_loss, validation_accuracy = self.test_global_model()
            self.export_data(r, training_loss, validation_loss, validation_accuracy)
        self.test_global_model(False)
        self.save_data()

    def initialize_clients(self):
        client_data_mapping = self.map_client_to_data()
        if self.algorithm == 'fedavg':
            return [FedAvgClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'scaffold':
            return [ScaffoldClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'fedproxy':
            return [FedProxyClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        elif self.algorithm == 'ifca':
            return [IFCAClient(index, self.dataset_name, client_data_mapping[index], self.batch_size, self.local_epochs) for index in range(self.n_clients)]
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def initialize_server(self):
        if self.algorithm == 'fedavg' or self.algorithm == 'fedproxy':
            return FedAvgServer(self.dataset_name)
        elif self.algorithm == 'scaffold':
            return ScaffoldServer(self.dataset_name)
        elif self.algorithm == 'ifca':
            return IFCAServer(self.dataset_name, self.number_of_clusters)
        else:
            raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def notify_clients(self):
        for client in self.clients:
            if self.algorithm == 'fedavg' or self.algorithm == 'fedproxy' or self.algorithm == 'ifca':
                client.notify_updates(self.server.model)
            elif self.algorithm == 'scaffold':
                client.notify_updates(self.server.model, self.server.control_state)
            else:
                raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')

    def clients_update(self):
        training_losses = []
        for client in self.clients:
            train_loss = client.train()
            training_losses.append(train_loss)
        average_training_loss = sum(training_losses) / len(training_losses)
        return average_training_loss

    def notify_server(self):
        client_data = {}
        for index, client in enumerate(self.clients):
            if self.algorithm == 'fedavg' or self.algorithm =='fedproxy' or self.algorithm == 'ifca':
                client_data[index] = client.model
            elif self.algorithm == 'scaffold':
                client_data[index] = { 'model': client.model, 'client_control_state': client.client_control_state }
            else:
                raise Exception(f'Algorithm {self.algorithm} not supported! Please check :)')
        self.server.receive_client_update(client_data)

    def server_update(self):
        self.server.aggregate()

    def initialize_data(self):
        d = self.get_dataset()
        dataset_size = len(d)
        training_size = int(dataset_size * 0.8)
        validation_size = dataset_size - training_size
        training_data, validation_data = random_split(d, [training_size, validation_size])
        return d, training_data, validation_data

    def map_client_to_data(self) -> dict[int, Subset]:
        clients_split = np.array_split(list(range(self.n_clients)), self.areas)
        mapping_area_clients = { areaId: list(clients_split[areaId]) for areaId in range(self.areas) }
        if self.partitioning.lower() == 'hard':
            mapping = utils.hard_non_iid_mapping(self.areas, len(self.complete_dataset.classes))
            distribution_per_area = utils.partitioning(mapping, self.training_data)
        elif self.partitioning.lower() == 'iid':
            mapping = utils.iid_mapping(self.areas, len(self.complete_dataset.classes))
            distribution_per_area = utils.partitioning(mapping, self.training_data)
        elif self.partitioning.lower() == 'dirichlet':
            distribution_per_area = utils.dirichlet_partitioning(self.training_data, self.areas, 0.5)
            self.save_distribution_heatmap(distribution_per_area)
        else:
            raise Exception(f'Partitioning {self.partitioning} not supported! Please check :)')
        
        mapping_client_data = self.__map_client_to_data(self.complete_dataset, mapping_area_clients, distribution_per_area)

        if self.algorithm == 'ifca':
            # mapping = utils.non_iid_mapping_validation(self.number_of_clusters, len(self.complete_dataset.classes))
            mapping = utils.non_iid_mapping_validation(self.areas, len(self.complete_dataset.classes))
            self.__validation_distribution_per_area = utils.partitioning(mapping, self.validation_data)
        return mapping_client_data

    def __map_client_to_data(self, dataset, mapping_area_clients, distribution_per_area):
        mapping_client_data = {}
        for area in mapping_area_clients.keys():
            clients = mapping_area_clients[area]
            indexes = distribution_per_area[area]
            random.shuffle(indexes)
            split = np.array_split(indexes, len(clients))
            for i, c in enumerate(clients):
                mapping_client_data[c] = Subset(dataset, split[i])
        return mapping_client_data

    def test_global_model(self, validation = True):
        if self.algorithm == 'ifca':
            if validation:
                validation_distribution = self.__validation_distribution_per_area
            else:
                dataset = self.get_dataset(False)
                mapping = utils.non_iid_mapping_validation(self.areas, len(dataset.classes))
                validation_distribution = utils.partitioning(mapping, dataset, True)

            losses = []
            accuracies = []
            models = self.server.model
            for i, model in enumerate(models):
                val_dataset = Subset(self.complete_dataset, validation_distribution[i])
                loss, accuracy = utils.test_model(model, val_dataset, self.batch_size, self.device)
                losses.append(loss)
                accuracies.append(accuracy)
            loss = sum(losses) / len(losses)
            accuracy = sum(accuracies) / len(accuracies)
            if not validation:
                data = pd.DataFrame({'Loss': [loss], 'Accuracy': [accuracy]})
                data.to_csv(f'{self.export_path}-test.csv', index=False)
            return loss, accuracy
        else:
            model = self.server.model
            if validation:
                dataset = self.validation_data
            else:
                dataset = self.get_dataset(False)
            loss, accuracy = utils.test_model(model, dataset, self.batch_size, self.device)
            if not validation:
                data = pd.DataFrame({'Loss': [loss], 'Accuracy': [accuracy]})
                data.to_csv(f'{self.export_path}-test.csv', index=False)
            return loss, accuracy

    def get_dataset(self, train = True):
        transform = transforms.Compose([transforms.ToTensor()])
        if self.dataset_name == 'MNIST':
            dataset = datasets.MNIST(root='dataset', train=train, download=True, transform=transform)
        elif self.dataset_name == 'CIFAR100':
            dataset = datasets.CIFAR100(root='dataset', train=train, download=True, transform=transform)
        elif self.dataset_name == 'EMNIST':
            dataset = datasets.EMNIST(root='dataset', split = 'letters', train=train, download=True, transform=transform)
        elif self.dataset_name == 'FashionMNIST':
            dataset = datasets.FashionMNIST(root='dataset', train=train, download=True, transform=transform)
        else:
            raise Exception(f'Dataset {self.dataset_name} not supported! Please check :)')
        return dataset

    def export_data(self, global_round, training_loss, evaluation_loss, evaluation_accuracy):
        """
        Registers new data, you can use it at each time stamp to store training and evaluation data.
        Important: it does not save the data on a file, you must call the specific method at the end of the simulation!
        :return: Nothing
        """
        self.simulation_data = self.simulation_data._append(
            {'Round': global_round,'TrainingLoss': training_loss, 'ValidationLoss': evaluation_loss, 'ValidationAccuracy': evaluation_accuracy},
            ignore_index=True
        )

    def save_data(self):
        """
        Saves the registered data on a file.
        :return: Nothing
        """
        self.simulation_data.to_csv(f'{self.export_path}.csv', index=False)

    def save_distribution_heatmap(self, distribution_per_area):
        matrix = []
        for k, indexes in distribution_per_area.items():
            v = [self.training_data.dataset.targets[index].item() for index in indexes]
            count = Counter(v)
            for i in range(len(self.training_data.dataset.classes)):
                if i not in count:
                    count[i] = 0
            count = dict(sorted(count.items()))
            matrix.append(count)

        rows = [[d[k] for k in d] for d in matrix]
        matrix = np.array(rows)
        utils.plot_heatmap(matrix, len(self.training_data.dataset.classes), self.areas, 'Dirichlet', False)
