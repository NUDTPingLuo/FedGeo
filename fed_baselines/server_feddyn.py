import copy
import torch
from fed_baselines.server_base import FedServer


class FedDynServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        self.alpha_dyn = 0.01

        # 初始化全局状态 h (仅针对可训练参数，不包含 BN 的 buffers)
        self.server_h = {
            name: torch.zeros_like(param.data, device='cpu')
            for name, param in self.model.named_parameters()
        }

    def agg(self):
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        old_global_model = self.model.state_dict()

        # 1. 更加安全的 avg_w 初始化逻辑
        avg_w = {}
        for key, val in old_global_model.items():
            if 'num_batches_tracked' in key:
                # 针对整型 buffer: 直接继承全局旧值，不置为0，也不参与后续浮点加权
                avg_w[key] = val.clone()
            else:
                # 针对浮点型 parameter 和 buffer (如 running_mean/var): 置为0准备累加
                avg_w[key] = torch.zeros_like(val)

        avg_loss = 0.0
        for name in self.selected_clients:
            weight = self.client_n_data[name] / self.n_data
            avg_loss += self.client_loss[name] * weight
            for key in avg_w:
                if 'num_batches_tracked' in key:
                    continue  # 完美避开 Float 转 Long 报错

                # 正常加权平均 (包含 weights 和 running_mean/var 等浮点 buffer)
                avg_w[key] += self.client_state[name][key].to(avg_w[key].device) * weight

        # 2. FedDyn 全局更新公式
        total_clients = len(self.client_list)

        # 关键修复：先让 new_model_state 继承所有已经完成加权平均的参数和 BN buffers
        new_model_state = copy.deepcopy(avg_w)

        # 然后，仅对可训练参数 (在 server_h 中) 应用 FedDyn 独有的纠偏逻辑
        for key in self.server_h:
            delta_w = avg_w[key] - old_global_model[key]

            # 更新全局对偶变量 h
            self.server_h[key] -= self.alpha_dyn * (client_num / total_clients) * delta_w.cpu()

            # 映射为最终模型: w_t+1 = avg_w - (1 / α) * h_t+1
            new_model_state[key] = avg_w[key] - (1.0 / self.alpha_dyn) * self.server_h[key].to(avg_w[key].device)

        self.model.load_state_dict(new_model_state)
        self.round += 1

        return new_model_state, avg_loss, self.n_data