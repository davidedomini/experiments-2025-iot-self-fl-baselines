import copy
from utils.FedUtils import *

class IFCAServer:

    def __init__(self, dataset_name, number_of_clusters):
        self.clients_data = {}
        self.dataset_name = dataset_name
        self.number_of_clusters = number_of_clusters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._models = [initialize_model(dataset_name).to(self.device) for _ in range(self.number_of_clusters)]

    def aggregate(self):
        # Find models for each cluster
        cluster_to_models = {}
        for _, (cluster_id, model) in self.clients_data.items():
            if cluster_id in cluster_to_models:
                cluster_to_models[cluster_id].append(model)
            else:
                cluster_to_models[cluster_id] = [model]

        # Aggregate models that are inside the same cluster
        for cluster_id, models in cluster_to_models.items():
            new_weights = self.__aggregate_inside_cluster(models)
            self._models[cluster_id].load_state_dict(new_weights)

    def receive_client_update(self, client_data):
        self.clients_data = client_data

    @property
    def model(self):
        return self._models

    def __aggregate_inside_cluster(self, models):
        w_avg = copy.deepcopy(models[0])
        for key in w_avg.keys():
            w_avg[key] = torch.mul(w_avg[key], 0.0)
        for key in w_avg.keys():
            for i in range(0, len(models)):
                w_avg[key] += models[i][key]
            w_avg[key] = torch.div(w_avg[key], len(models))
        return w_avg