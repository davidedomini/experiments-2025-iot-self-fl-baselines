import copy
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from utils.FedUtils import initialize_model, test_model

class IFCAClient:

    def __init__(self, mid, dataset_name, dataset, batch_size, epochs):
        self.mid = mid
        self.epochs = epochs
        self.dataset = dataset
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._global_models = []
        self.current_cluster_id = 0

    def train(self):
        self.current_cluster_id = self.__find_cluster()
        model = self._global_models[self.current_cluster_id]
        train_loader = DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_func = nn.CrossEntropyLoss()
        losses = []
        model.to(self.device)
        for _ in range(self.epochs):
            batch_losses = []
            for step, (images, labels) in enumerate(train_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.enable_grad():
                    model.train()
                    outputs = model(images)
                    loss = loss_func(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch_losses.append(loss.item())
            mean_epoch_loss = sum(batch_losses) / len(batch_losses)
            losses.append(mean_epoch_loss)
        return sum(losses) / len(losses)

    def notify_updates(self, global_models):
        for global_model in global_models:
            fresh_model = copy.deepcopy(global_model)
            self._global_models.append(fresh_model)

    def __find_cluster(self):
        losses = []
        for global_model in self._global_models:
            loss, _ = test_model(global_model, self.dataset, self.batch_size, self.device)
            losses.append(loss)
        cluster_id, _ = min(enumerate(numeri), key=lambda x: x[1])
        return cluster_id

    @property
    def model(self):
        return self.current_cluster_id, self._global_models[self.current_cluster_id]