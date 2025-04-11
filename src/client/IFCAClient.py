import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils.FedUtils import initialize_model, test_model

class IFCAClient:

    def __init__(self, mid, dataset_name, dataset, batch_size, epochs):
        self.mid = mid
        self.lr = 0.001
        self.epochs = epochs
        self.dataset = dataset
        # labels = [self.dataset[idx][1] for idx in range(len(self.dataset))]
        # print(f'Client {self.mid} --> training set size {len(self.dataset)} classes {set(labels)}')
        self._global_models = []
        self.weight_decay = 1e-4
        self.current_cluster_id = 0
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self._model = initialize_model(dataset_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

    def train(self):
        self.current_cluster_id = self.__find_cluster()
        self._model.load_state_dict(self._global_models[self.current_cluster_id].state_dict())
        train_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = nn.CrossEntropyLoss()
        losses = []
        self._model.to(self.device)
        for _ in range(self.epochs):
            batch_losses = []
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.enable_grad():
                    self._model.train()
                    outputs = self._model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
            mean_epoch_loss = sum(batch_losses) / len(batch_losses)
            losses.append(mean_epoch_loss)
        return sum(losses) / len(losses)

    def notify_updates(self, global_models):
        self._global_models = []
        for global_model in global_models:
            fresh_model = copy.deepcopy(global_model)
            self._global_models.append(fresh_model)

    def __find_cluster(self):
        losses = []
        for global_model in self._global_models:
            loss, _ = test_model(global_model, self.dataset, self.batch_size, self.device)
            losses.append(loss)
        cluster_id, _ = min(enumerate(losses), key=lambda x: x[1])
        return cluster_id

    @property
    def model(self):
        return self.current_cluster_id, self._model