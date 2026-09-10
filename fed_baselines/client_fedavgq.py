"""FedAvg with the exact same incremental quantizer used by FedGeo."""

import torch
from torch.utils.data import DataLoader

from fed_baselines.client_fedgeo import GeoClient


class FedAvgQClient(GeoClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum, q_bits):
        super().__init__(
            name, epoch, dataset_id, model_name, lr, batch_size, momentum,
            beta=0.0, q_bits=q_bits
        )

    def collect_grads(self, normal=True):
        """Run ordinary local SGD; ``normal`` is accepted for API compatibility."""
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )
        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self._lr, momentum=self._momentum
        )
        loss_func = torch.nn.CrossEntropyLoss()
        last_loss = 0.0

        for _ in range(self._epoch):
            for x, y in train_loader:
                x = x.to(self._device, non_blocking=True)
                y = y.to(self._device, non_blocking=True)
                self.model.train()
                optimizer.zero_grad()
                loss = loss_func(self.model(x), y.long())
                loss.backward()
                optimizer.step()
                last_loss = loss.item()

        return last_loss
