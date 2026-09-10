import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from fed_baselines.client_base import FedClient


class FedSAMClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        # 领域扰动半径 rho，通常设为 0.05
        self.rho = 0.01

    def train(self):
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = nn.CrossEntropyLoss()

        for epoch in range(self._epoch):
            for x, y in train_loader:
                b_x, b_y = x.to(self._device), y.to(self._device)

                # ==========================================================
                # 步骤 1：计算正常梯度
                # ==========================================================
                self.model.train()
                optimizer.zero_grad()
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())
                loss.backward()

                # ==========================================================
                # 步骤 2：计算并注入扰动 (w_adv = w + e_w)
                # ==========================================================
                grad_norm = torch.norm(
                    torch.stack([p.grad.norm(p=2) for p in self.model.parameters() if p.grad is not None]),
                    p=2
                )
                scale = self.rho / (grad_norm + 1e-12)

                e_w_dict = {}
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            e_w = p.grad * scale
                            p.add_(e_w)  # 移动到恶劣位置
                            e_w_dict[name] = e_w

                # ==========================================================
                # 步骤 3：在扰动位置上重新计算梯度
                # ==========================================================
                optimizer.zero_grad()
                output_adv = self.model(b_x)
                loss_adv = loss_func(output_adv, b_y.long())
                loss_adv.backward()

                # ==========================================================
                # 步骤 4：恢复原始权重，并用新梯度更新
                # ==========================================================
                with torch.no_grad():
                    for name, p in self.model.named_parameters():
                        if p.grad is not None:
                            p.sub_(e_w_dict[name])  # 退回原始位置 w

                optimizer.step()

        return self.model.state_dict(), self.n_data, loss_adv.item()
