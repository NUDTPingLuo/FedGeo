"""Client implementation of FedCM (Xu et al., 2021)."""

import torch
from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient


class FedCMClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size,
                 momentum, cm_alpha=0.1):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        if not 0.0 < cm_alpha <= 1.0:
            raise ValueError("fedcm_alpha must be in (0, 1]")
        self.cm_alpha = float(cm_alpha)
        self.server_direction = {}

    def update(self, total_round, global_round, model_state_dict, server_direction):
        super().update(total_round, global_round, model_state_dict)
        self.server_direction = {
            key: value.to(self._device) for key, value in server_direction.items()
        }

    def train(self):
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )
        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self._lr, momentum=self._momentum
        )
        loss_func = torch.nn.CrossEntropyLoss()
        local_steps = 0
        last_loss = 0.0

        for _ in range(self._epoch):
            for x, y in train_loader:
                x = x.to(self._device, non_blocking=True)
                y = y.to(self._device, non_blocking=True)
                self.model.train()
                optimizer.zero_grad()
                loss = loss_func(self.model(x), y.long())
                loss.backward()

                with torch.no_grad():
                    for name, parameter in self.model.named_parameters():
                        if parameter.grad is None:
                            continue
                        direction = self.server_direction.get(name)
                        parameter.grad.mul_(self.cm_alpha)
                        if direction is not None:
                            parameter.grad.add_(direction, alpha=1.0 - self.cm_alpha)

                optimizer.step()
                local_steps += 1
                last_loss = loss.item()

        return self.model.state_dict(), self.n_data, last_loss, local_steps, self._lr
