import torch
import math
from fed_baselines.client_base import FedClient
from torch.utils.data import DataLoader
from utils.quantization import dequantize_tensor, payload_nbytes, quantize_tensor


class GeoClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum, beta, q_bits):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        self._beta = beta
        self.init_beta = beta
        self.init_lr = lr
        self.q_bits = q_bits
        self.global_rectify_gradient = None
        self.initial_state_dict = None
        self.synced_global_model = None

    @torch.no_grad()
    def update(self, total_round, global_round, server_payload):
        eta_min = 0.0
        self._lr = eta_min + 0.5 * (self.init_lr - eta_min) * (
                1 + math.cos(math.pi * global_round / total_round)
        )
        self.global_round = global_round
        self.global_rectify_gradient = {}

        if server_payload["type"] == "full_model":
            self.model.load_state_dict(server_payload["data"])
            self.synced_global_model = {k: v.clone().cpu() for k, v in server_payload["data"].items()}

        elif server_payload["type"] == "q_grad":
            quantized_data = server_payload["data"]
            bn_stats = server_payload["bn_stats"]

            for key in quantized_data:
                dense_val = dequantize_tensor(quantized_data[key], self._device)
                self.global_rectify_gradient[key] = dense_val

            for key in self.synced_global_model:
                if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                    self.synced_global_model[key].copy_(bn_stats[key].cpu())
                else:
                    self.synced_global_model[key].sub_(self.global_rectify_gradient[key].cpu())

            self.model.load_state_dict(self.synced_global_model)

        self.initial_state_dict = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}

    def collect_grads(self, normal=True):
        train_loader = DataLoader(
            self.trainset, batch_size=self._batch_size, shuffle=True,
            num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False
        )

        self.model.to(self._device)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        loss_func = torch.nn.CrossEntropyLoss()

        if not normal:
            self.active_params = []
            self.g_stars_list = []

            for name, p in self.model.named_parameters():
                if p.requires_grad and name in self.global_rectify_gradient:
                    self.active_params.append(p)
                    self.g_stars_list.append(self.global_rectify_gradient[name])

            if self.g_stars_list:
                g_star_norm = torch.norm(torch.stack([g.norm() for g in self.g_stars_list])) + 1e-12
            else:
                g_star_norm = torch.tensor(1e-12, device=self._device)

            # 🔥 极速魔法 1：保留一个张量形式的 norm，用于循环内的纯 GPU 异步计算
            g_star_norm_tensor = g_star_norm.clone().detach()

            # 只有 alpha 需要 Python float，由于它在循环外计算，这里的 .item() 毫无额外开销
            factor_g_star_val = (self._beta / g_star_norm).item()

        for _ in range(self._epoch):
            for x, y in train_loader:
                # 🔥 极速魔法 2：异步数据拷贝，不再让 GPU 等待内存拷贝
                b_x = x.to(self._device, non_blocking=True)
                b_y = y.to(self._device, non_blocking=True)

                self.model.train()
                optimizer.zero_grad()
                output = self.model(b_x)
                loss = loss_func(output, b_y.long())

                if normal:
                    loss.backward()
                    optimizer.step()
                else:
                    loss.backward()

                    with torch.no_grad():
                        local_grads = [p.grad for p in self.active_params]

                        dot_val = torch.sum(
                            torch.stack([torch.sum(g * g_star) for g, g_star in zip(local_grads, self.g_stars_list)]))
                        local_norm = torch.norm(torch.stack([g.norm() for g in local_grads])) + 1e-12

                        # ================================================================
                        # 🔥 极速魔法 3：彻底消灭 .item() 同步阻塞！
                        # 让 cos_val 和 factor_g 保持为 GPU 上的 0D 张量 (0-dimensional Tensor)
                        # CPU 瞬间下达指令后直接放行，GPU 在后台流水线里慢慢算，互不干扰！
                        # ================================================================
                        cos_val = torch.clamp(dot_val / (local_norm * g_star_norm_tensor), -1.0, 1.0)
                        factor_g_tensor = 1.0 - self._beta * (cos_val / local_norm)

                        for g, g_star in zip(local_grads, self.g_stars_list):
                            # mul_ 完美支持传入 GPU 张量，无需变回 Python 数值！
                            g.mul_(factor_g_tensor).add_(g_star, alpha=factor_g_star_val)

                    optimizer.step()

        return loss.item()

    # 🚀 移除了错误的 @torch.no_grad() 装饰器！
    def train(self):
        normal = (self.global_round == 0)
        # 现在 collect_grads 能够正常生成带有梯度的 loss 并反向传播了
        loss = self.collect_grads(normal=normal)

        # 🔥 只在计算伪梯度和量化时开启 no_grad，杜绝计算图积累
        with torch.no_grad():
            pseudo_grad = {}
            bn_stats = {}
            model_state = self.model.state_dict()

            for name in model_state:
                if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
                    bn_stats[name] = model_state[name].cpu()
                else:
                    pseudo_grad[name] = self.initial_state_dict[name] - model_state[name].cpu()

            payload = {}

            if self.q_bits >= 32:
                payload["type"] = "full"
                payload["data"] = pseudo_grad
                payload["bn_stats"] = bn_stats
            else:
                payload["type"] = "qsgd"
                payload["data"] = {
                    key: quantize_tensor(value.to(self._device), self.q_bits)
                    for key, value in pseudo_grad.items()
                }
                payload["bn_stats"] = bn_stats

            payload["wire_bytes"] = payload_nbytes(payload)

        return payload, self.n_data, loss
