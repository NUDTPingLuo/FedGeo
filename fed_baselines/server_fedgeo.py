from fed_baselines.server_base import FedServer
import copy
import torch
import math
import os
import csv
from utils.quantization import dequantize_tensor, payload_nbytes, quantize_tensor


def calculate_payload_mb(data_obj):
    return payload_nbytes(data_obj) / (1024 * 1024)


class GeoServer(FedServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size, alpha,
                 q_bits=32, run_tag=None):
        super().__init__(client_list, dataset_id, model_name, batch_size)
        self.prev_global_grad = None
        self.q_bits = q_bits
        self.alpha = alpha  # 保存 alpha 用于命名
        self.last_uplink_bytes = 0
        self.last_downlink_bytes = 0
        os.makedirs("cos_logs", exist_ok=True)

        if run_tag is None:
            self.csv_filename = os.path.join(
                "cos_logs", f"{dataset_id}_alpha{alpha}_cos_matrix.csv"
            )
            self.drift_csv_filename = os.path.join(
                "cos_logs", f"FedGeo_{dataset_id}_alpha{alpha}_drift_cancellation.csv"
            )
        else:
            self.csv_filename = os.path.join("cos_logs", run_tag + "_cos_matrix.csv")
            self.drift_csv_filename = os.path.join(
                "cos_logs", run_tag + "_drift_cancellation.csv"
            )

    def agg(self):
        client_num = len(self.selected_clients)
        if client_num == 0 or self.n_data == 0:
            return self.model.state_dict(), 0, 0

        self.model.to(self._device)
        global_model = self.model.state_dict()
        new_model = copy.deepcopy(global_model)
        avg_loss = 0

        weighted_avg_grad = {}
        weighted_bn_stats = {}
        round_cos_dict = {}

        # ====================================================
        # 🚀 理论验证跟踪变量
        # ====================================================
        expected_client_norm_sq = 0.0  # E[||g_k||^2]
        expected_cancel_base = 0.0  # E[ ||g_global|| * sin^2(theta_k) ]

        # 1. 接收客户端的量化伪梯度并加权聚合
        for i, name in enumerate(self.selected_clients):
            weight = self.client_n_data[name] / self.n_data
            payload = self.client_state[name]["payload"]
            bn_stats = payload["bn_stats"]
            payload_bytes = (
                payload["wire_bytes"] if "wire_bytes" in payload else payload_nbytes(payload)
            )
            self.last_uplink_bytes += int(payload_bytes)

            for key in bn_stats:
                bn_value = bn_stats[key].to(self._device)
                if not torch.is_floating_point(bn_value):
                    if key not in weighted_bn_stats:
                        weighted_bn_stats[key] = bn_value.clone()
                elif key not in weighted_bn_stats:
                    weighted_bn_stats[key] = bn_value * weight
                else:
                    weighted_bn_stats[key] += bn_value * weight

            client_dot = 0.0
            client_norm_sq = 0.0
            global_norm_sq = 0.0

            if payload["type"] == "full":
                client_grad = payload["data"]
                for key in client_grad:
                    tensor_data = client_grad[key].to(self._device)
                    if i == 0:
                        weighted_avg_grad[key] = tensor_data * weight
                    else:
                        weighted_avg_grad[key] += tensor_data * weight

                    if self.round > 0 and self.prev_global_grad is not None:
                        prev_g = self.prev_global_grad[key].to(self._device)
                        client_dot += torch.sum(tensor_data * prev_g).item()
                        client_norm_sq += torch.sum(tensor_data * tensor_data).item()
                        global_norm_sq += torch.sum(prev_g * prev_g).item()

            elif payload["type"] == "qsgd":
                quantized_data = payload["data"]
                for key in quantized_data:
                    reconstructed_grad = dequantize_tensor(quantized_data[key], self._device)

                    if i == 0:
                        weighted_avg_grad[key] = reconstructed_grad * weight
                    else:
                        weighted_avg_grad[key] += reconstructed_grad * weight

                    if self.round > 0 and self.prev_global_grad is not None:
                        prev_g = self.prev_global_grad[key].to(self._device)
                        client_dot += torch.sum(reconstructed_grad * prev_g).item()
                        client_norm_sq += torch.sum(reconstructed_grad * reconstructed_grad).item()
                        global_norm_sq += torch.sum(prev_g * prev_g).item()

            # 🚀 计算 B^2 的第一步：累加加权后的客户端梯度模长平方 E[||g_k||^2]
            expected_client_norm_sq += client_norm_sq * weight

            if self.round > 0 and self.prev_global_grad is not None:
                cos_val = client_dot / (math.sqrt(client_norm_sq) * math.sqrt(global_norm_sq) + 1e-12)
                cos_val = max(min(cos_val, 1.0), -1.0)
                round_cos_dict[name] = cos_val

                # 🚀 计算几何纠偏力 Cancel_Base: ||g_global|| * sin^2(theta)
                sin_sq = 1.0 - cos_val ** 2
                expected_cancel_base += (math.sqrt(global_norm_sq) * max(0.0, sin_sq)) * weight

            avg_loss += self.client_loss[name] * weight

        # ====================================================
        # 🚀 计算 B^2 的第二步与日志写入
        # ====================================================
        # B^2 = E[||g_k||^2] - ||E[g_k]||^2
        current_global_norm_sq = sum(torch.sum(v * v).item() for v in weighted_avg_grad.values())
        B_sq = max(0.0, expected_client_norm_sq - current_global_norm_sq)

        if self.round > 0 and round_cos_dict:
            # 1. 记录漂移对冲日志 (Drift vs Cancellation)
            if self.round == 1:
                with open(self.drift_csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Round", "B_sq", "Cancel_Base"])

            with open(self.drift_csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.round, B_sq, expected_cancel_base])

            # 2. 记录 Cos 动力学矩阵 (原逻辑)
            fixed_client_list = sorted(self.client_list)
            if self.round == 1:
                with open(self.csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    header = ["Round"] + [str(name) for name in fixed_client_list]
                    writer.writerow(header)

            with open(self.csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                row = [self.round] + [round_cos_dict.get(name, 0.0) for name in fixed_client_list]
                writer.writerow(row)

        # ====================================================
        # 2. 对聚合出的全局梯度进行下行量化压缩
        # ====================================================
        quantized_global_grad = {}
        reconstructed_global_grad = {}

        for key in weighted_avg_grad:
            target_tensor = weighted_avg_grad[key]
            quantized_global_grad[key] = quantize_tensor(target_tensor, self.q_bits)
            # 服务端也用解码后的梯度更新，确保与客户端严格同步。
            reconstructed_global_grad[key] = dequantize_tensor(
                quantized_global_grad[key], self._device
            )

        # 生成下发给客户端的数据包
        server_payload = {
            "type": "q_grad",
            "data": quantized_global_grad,
            "bn_stats": weighted_bn_stats
        }
        server_payload["wire_bytes"] = payload_nbytes(server_payload)
        self.last_downlink_bytes = int(server_payload["wire_bytes"])

        # ====================================================
        # 3. 使用【携带量化噪声的还原梯度】更新服务端的全局模型
        # ====================================================
        self.prev_global_grad = {k: v.clone().cpu() for k, v in reconstructed_global_grad.items()}

        for key in global_model:
            if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
                if key in weighted_bn_stats:
                    new_model[key] = weighted_bn_stats[key]
            else:
                new_model[key] = global_model[key] - reconstructed_global_grad[key]

        self.model.load_state_dict(new_model)
        self.round += 1

        # 返回 payload 包，不再返回庞大的模型字典
        return server_payload, avg_loss, self.n_data

    def rec(self, name, payload, n_data, loss):
        self.n_data += n_data
        self.client_state[name] = {"payload": payload}
        self.client_n_data[name] = n_data
        self.client_loss[name] = loss

    def flush(self):
        self.n_data = 0
        self.client_state = {}
        self.client_n_data = {}
        self.client_loss = {}
        self.last_uplink_bytes = 0
