import copy
import torch
from torch import nn
from utils.FedUtils import initialize_model
from torch.utils.data import DataLoader, random_split

class FedProxyClient:

    def __init__(self, mid, dataset_name, dataset, batch_size, epochs):
        self.mu = 0.1
        self.mid = mid
        self.lr = 0.001
        self.epochs = epochs
        self.weight_decay=1e-4
        self.training_set = dataset
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = initialize_model(dataset_name).to(self.device)

    def train(self):
        # labels = [self.training_set[idx][1] for idx in range(len(self.training_set))]
        # print(f'Client {self.mid} --> training set size {len(self.training_set)} classes {set(labels)}')

        global_weights = copy.deepcopy(self._model.state_dict()) # w^t
        global_model = initialize_model(self.dataset_name)
        global_model.load_state_dict(global_weights)
        global_model.to(self.device)
        global_weights = list(global_model.parameters())

        train_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = nn.CrossEntropyLoss()
        losses = []
        self.model.to(self.device)
        for epoch in range(self.epochs):
            batch_losses = []
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.enable_grad():
                    self._model.train()
                    outputs = self._model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss = loss + self.proximal_term(self._model, global_weights)
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
            mean_epoch_loss = sum(batch_losses) / len(batch_losses)
            losses.append(mean_epoch_loss)
        return sum(losses) / len(losses)

    def proximal_term(self, trained_model, server_model):
        prox_term = 0.0
        for p_i, param in enumerate(trained_model.parameters()):
            prox_term += (self.mu / 2) * torch.norm((param - server_model[p_i])) ** 2
        return prox_term

    def notify_updates(self, global_model):
        self._model.load_state_dict(copy.deepcopy(global_model.state_dict()))

    @property
    def model(self):
        return self._model