import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient


class FedDynClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        # FedDyn 的正则化强度系数，通常设为 0.01 或 0.1
        self.alpha_dyn = 0.01
        self.local_h = None

    def train(self):
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )
        self.model.to(self._device)
        self.model.train()

        # 初始化 local_h (第一次参与训练时)
        if self.local_h is None:
            self.local_h = {
                name: torch.zeros_like(param.data, device='cpu')
                for name, param in self.model.named_parameters()
            }

        # 冻结并保存下发的全局模型 w_t
        global_weights = {
            name: param.detach().clone().to(self._device)
            for name, param in self.model.named_parameters()
        }

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(self._epoch):
            for x, y in train_loader:
                b_x, b_y = x.to(self._device), y.to(self._device)

                optimizer.zero_grad()
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()

                # ===== FedDyn 核心：修改梯度 =====
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is None:
                            continue
                        # 1. Proximal term: α * (w - w_t)
                        prox_term = self.alpha_dyn * (p.data - global_weights[name])
                        # 2. Dynamic penalty term: - h_i
                        dyn_term = -self.local_h[name].to(self._device)

                        # 直接把惩罚项加到梯度里，交给 optimizer 去 step
                        p.grad.data.add_(prox_term + dyn_term)

                optimizer.step()

        # ===== 训练结束后，更新本地状态 local_h =====
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                delta_w = p.data.detach() - global_weights[name]
                self.local_h[name] -= self.alpha_dyn * delta_w.to('cpu')

        return self.model.state_dict(), self.n_data, loss.item()
