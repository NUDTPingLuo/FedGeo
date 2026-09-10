"""Fixed-radius FedSOL client following the official two-step update."""

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fed_baselines.client_base import FedClient


class FedSOLClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size,
                 momentum, rho=0.1, temperature=3.0):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        if rho < 0:
            raise ValueError("fedsol_rho must be non-negative")
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.global_model = None

    def update(self, total_round, global_round, model_state_dict):
        super().update(total_round, global_round, model_state_dict)
        self.global_model = copy.deepcopy(self.model).to(self._device)
        self.global_model.eval()
        for parameter in self.global_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _set_batchnorm_momentum(model, momentum):
        previous = []
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                previous.append((module, module.momentum))
                module.momentum = momentum
        return previous

    def train(self):
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )
        self.model.to(self._device)
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self._lr, momentum=self._momentum
        )
        criterion = torch.nn.CrossEntropyLoss()
        last_loss = 0.0

        for _ in range(self._epoch):
            for x, y in train_loader:
                x = x.to(self._device, non_blocking=True)
                y = y.to(self._device, non_blocking=True)
                self.model.train()

                # First pass: output-space proximal restriction relative to the
                # frozen downloaded global model (official FedSOL fixed form).
                optimizer.zero_grad()
                local_logits = self.model(x)
                with torch.no_grad():
                    global_probs = F.softmax(self.global_model(x) / self.temperature, dim=1)
                proximal_loss = F.kl_div(
                    F.log_softmax(local_logits / self.temperature, dim=1),
                    global_probs,
                    reduction="batchmean"
                )
                proximal_loss.backward()

                grad_norms = [
                    parameter.grad.norm(p=2)
                    for parameter in self.model.parameters()
                    if parameter.grad is not None
                ]
                if grad_norms:
                    grad_norm = torch.norm(torch.stack(grad_norms), p=2)
                else:
                    grad_norm = torch.tensor(0.0, device=self._device)

                perturbations = []
                scale = self.rho / (grad_norm + 1e-12)
                with torch.no_grad():
                    for parameter in self.model.parameters():
                        perturbation = (
                            torch.zeros_like(parameter) if parameter.grad is None
                            else parameter.grad * scale
                        )
                        parameter.add_(perturbation)
                        perturbations.append(perturbation)

                # Second pass: task gradient at the perturbed point.  Freeze BN
                # running statistics here so each batch updates them only once.
                optimizer.zero_grad()
                bn_state = self._set_batchnorm_momentum(self.model, 0.0)
                loss = criterion(self.model(x), y.long())
                loss.backward()
                for module, momentum in bn_state:
                    module.momentum = momentum

                with torch.no_grad():
                    for parameter, perturbation in zip(self.model.parameters(), perturbations):
                        parameter.sub_(perturbation)
                optimizer.step()
                last_loss = loss.item()

        return self.model.state_dict(), self.n_data, last_loss
