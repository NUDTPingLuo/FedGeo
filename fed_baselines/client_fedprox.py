from fed_baselines.client_base import FedClient
import copy
from utils.models import *

from torch.utils.data import DataLoader


class FedProxClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self.mu = 0.01

    def train(self):
        train_loader = DataLoader(
            self.trainset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )

        self.model.to(self._device)
        self.model.train()

        # ===== 固定 global model =====
        global_weights = [
            p.detach().clone().to(self._device)
            for p in self.model.parameters()
        ]

        loss_func = nn.CrossEntropyLoss()

        for epoch in range(self._epoch):
            for x, y in train_loader:
                b_x = x.to(self._device)
                b_y = y.to(self._device)

                # ===== 1. 清梯度 =====
                self.model.zero_grad()

                # ===== 2. 正常反传 =====
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()

                # ===== 3. 显式 FedProx 更新（batch-level）=====
                with torch.no_grad():
                    for i, p in enumerate(self.model.parameters()):
                        if p.grad is None:
                            continue

                        # ∇ℓ_k(w)
                        grad = p.grad

                        # μ (w − w_global)
                        prox_grad = self.mu * (p.data - global_weights[i])

                        # 显式 SGD / FedProx
                        # The proximal term is part of the gradient and must be
                        # scaled by the learning rate as well.
                        p.data -= self._lr * (grad + prox_grad)

        return self.model.state_dict(), self.n_data, loss.item()
