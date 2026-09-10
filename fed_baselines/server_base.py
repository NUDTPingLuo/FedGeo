from utils.models import *
import torch
from torch.utils.data import DataLoader
from utils.fed_utils import assign_dataset, init_model
from utils.quantization import state_dict_nbytes
import os


class FedServer(object):
    def __init__(self, client_list, dataset_id, model_name, batch_size):
        """
        Initialize the server for federated learning.
        :param client_list: List of the connected clients in networks
        :param dataset_id: Dataset name for the application scenario
        :param model_name: Machine learning model name for the application scenario
        """
        # Initialize the dict and list for system settings
        self.client_state = {}
        self.client_loss = {}
        self.client_n_data = {}
        self.selected_clients = []
        # self._lr = 0.01
        # batch size for testing
        self._batch_size = batch_size
        self.client_list = client_list
        self._num_workers = 0 if os.name == "nt" else 4
        self.last_uplink_bytes = 0
        self.last_downlink_bytes = 0

        # Initialize the test dataset
        self.testset = None

        # Initialize the hyperparameter for federated learning in the server
        self.round = 0
        self.n_data = 0
        self._dataset_id = dataset_id

        # Testing on GPU
        gpu = 0
        self._device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() and gpu != -1 else "cpu")

        # Initialize the global machine learning model
        self._num_class, self._image_dim, self._image_channel = assign_dataset(dataset_id)
        self.model_name = model_name
        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)

    def load_testset(self, testset):
        """
        Server loads the test dataset.
        :param data: Dataset for testing.
        """
        self.testset = testset

    def state_dict(self):
        """
        Server returns global model dict.
        :return: Global model dict
        """
        return self.model.state_dict()

    def test(self):
        """
        Server tests the model on test dataset.
        """
        test_loader = DataLoader(self.testset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)
        self.model.to(self._device)
        self.model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self._device), y.to(self._device)
                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y).sum().item()
                total += y.size(0)

        accuracy = correct / total
        return accuracy

    def select_clients(self, connection_ratio=1):
        """
        Server selects a fraction of clients.
        :param connection_ratio: connection ratio in the clients
        """
        # select a fraction of clients
        self.selected_clients = []
        self.n_data = 0
        for client_id in self.client_list:
            b = np.random.binomial(np.ones(1).astype(int), connection_ratio)
            if b:
                self.selected_clients.append(client_id)
                self.n_data += self.client_n_data[client_id]

    def agg(self):
        """
        Server aggregates models from connected clients.
        :return: model_state: Updated global model after aggregation
        :return: avg_loss: Averaged loss value
        :return: n_data: Number of the local data points
        """
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        # Initialize a model for aggregation
        model_state = {}
        avg_loss = 0

        # Aggregate the local updated models from selected clients
        for name in self.selected_clients:
            if name not in self.client_state:
                continue
            weight = self.client_n_data[name] / self.n_data
            for key, value in self.client_state[name].items():
                # Integer buffers such as BatchNorm.num_batches_tracked must
                # not be multiplied by floating-point aggregation weights.
                if not torch.is_floating_point(value):
                    if key not in model_state:
                        model_state[key] = value.clone()
                elif key not in model_state:
                    model_state[key] = value * weight
                else:
                    model_state[key] = model_state[key] + value * weight

            avg_loss = avg_loss + self.client_loss[name] * self.client_n_data[name] / self.n_data

        # Server load the aggregated model as the global model
        self.model.load_state_dict(model_state)
        self.round = self.round + 1
        n_data = self.n_data
        model_bytes = state_dict_nbytes(model_state)
        self.last_uplink_bytes = model_bytes * client_num
        self.last_downlink_bytes = model_bytes * client_num

        return model_state, avg_loss, n_data

    def rec(self, name, state_dict, n_data, loss):
        """
        Server receives the local updates from the connected client k.
        :param name: Name of client k
        :param state_dict: Model dict from the client k
        :param n_data: Number of local data points in the client k
        :param loss: Loss of local training in the client k
        """
        self.n_data = self.n_data + n_data
        self.client_state[name] = {}
        self.client_n_data[name] = {}

        self.client_state[name].update(state_dict)
        self.client_n_data[name] = n_data
        self.client_loss[name] = {}
        self.client_loss[name] = loss

    def flush(self):
        """
        Flushing the client information in the server
        """
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
