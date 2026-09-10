"""Server state and aggregation for FedCM."""

import copy

import torch

from fed_baselines.server_base import FedServer
from utils.quantization import state_dict_nbytes


class FedCMServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        self.client_steps = {}
        self.client_lr = {}
        self.global_direction = {
            name: torch.zeros_like(parameter, device="cpu")
            for name, parameter in self.model.named_parameters()
        }
        self.last_uplink_bytes = 0
        self.last_downlink_bytes = 0

    def direction_state_dict(self):
        return self.global_direction

    def rec(self, name, state_dict, n_data, loss, local_steps, lr):
        super().rec(name, state_dict, n_data, loss)
        self.client_steps[name] = max(1, int(local_steps))
        self.client_lr[name] = float(lr)

    def agg(self):
        if not self.selected_clients or self.n_data == 0:
            return self.model.state_dict(), 0.0, 0, self.global_direction

        old_state = copy.deepcopy(self.model.state_dict())
        new_state = copy.deepcopy(old_state)
        avg_loss = 0.0
        effective_lr_steps = 0.0

        for index, name in enumerate(self.selected_clients):
            weight = self.client_n_data[name] / self.n_data
            client_state = self.client_state[name]
            for key, value in client_state.items():
                if not torch.is_floating_point(value):
                    if index == 0:
                        new_state[key] = value.clone()
                elif index == 0:
                    new_state[key] = value * weight
                else:
                    new_state[key] += value * weight
            avg_loss += self.client_loss[name] * weight
            effective_lr_steps += self.client_lr[name] * self.client_steps[name] * weight

        denominator = max(effective_lr_steps, 1e-12)
        parameter_names = set(name for name, _ in self.model.named_parameters())
        self.global_direction = {
            key: ((old_state[key].cpu() - new_state[key].cpu()) / denominator)
            for key in parameter_names
        }
        self.model.load_state_dict(new_state)
        self.round += 1
        model_bytes = state_dict_nbytes(new_state)
        direction_bytes = state_dict_nbytes(self.global_direction)
        self.last_uplink_bytes = model_bytes * len(self.selected_clients)
        self.last_downlink_bytes = (model_bytes + direction_bytes) * len(self.selected_clients)
        return new_state, avg_loss, self.n_data, self.global_direction

    def flush(self):
        super().flush()
        self.client_steps = {}
        self.client_lr = {}

