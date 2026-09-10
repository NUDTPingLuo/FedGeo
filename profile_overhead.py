#!/usr/bin/env python
import os
import random
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

# ===== 导入所有的 Client 和 Server =====
from fed_baselines.client_base import FedClient
from fed_baselines.client_fedgeo import GeoClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_feddyn import FedDynClient
from fed_baselines.client_fedsam import FedSAMClient
from fed_baselines.client_fedcm import FedCMClient
from fed_baselines.client_fedsol import FedSOLClient

from fed_baselines.server_base import FedServer
from fed_baselines.server_fedgeo import GeoServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fedcm import FedCMServer

from preprocessing.baselines_dataloader import divide_data_dirichlet_resample


def profile_client(client, client_type, num_runs=30):
    """
    使用高精度 GPU 时钟测量客户端单次完整训练的耗时
    """
    # 预热 5 次，消除 GPU 冷启动、显存分配、和首次寻址的开销干扰
    for _ in range(5):
        client.train()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    time_records = []

    for _ in tqdm(range(num_runs), desc=f"Profiling {client_type:<10}", leave=False):
        start_event.record()

        # 核心：执行客户端的完整本地训练逻辑 (兼容所有算法的不同返回值)
        _ = client.train()

        end_event.record()
        # ⚠️ 必须同步！强制 CPU 等待 GPU 队列中的所有指令执行完毕
        torch.cuda.synchronize()

        time_ms = start_event.elapsed_time(end_event)
        time_records.append(time_ms)

    avg_time = np.mean(time_records)
    std_time = np.std(time_records)

    return avg_time, std_time


def run_profiling():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')
    parser.add_argument('--runs', type=int, default=30, help='Number of profiling runs')
    args = parser.parse_args()

    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # 固定随机种子保证公平对比
    torch.manual_seed(42)
    np.random.seed(42)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("=" * 70)
    print(f"🚀 端侧计算开销全量评估 (Device: {device_name})")
    print(f"Dataset: {config['system']['dataset']} | Model: {config['system']['model']}")
    print(f"Epoch: {config['client']['num_local_epoch']} | Batch Size: {config['client']['batch_size']}")
    print("=" * 70)

    # 划分数据
    # 划分数据
    trainset_config, testset = divide_data_dirichlet_resample(
        num_client=config["system"]["num_client"],  # <--- 动态读取真实的客户端总数
        alpha=config["system"]["dirichlet_alpha"],
        dataset_name=config["system"]["dataset"],
        i_seed=42
    )

    test_client_id = trainset_config['users'][0]
    client_data = trainset_config['user_data'][test_client_id]

    # 用于提供全局权重的通用 Server
    base_server = FedServer([test_client_id], dataset_id=config["system"]["dataset"],
                            model_name=config["system"]["model"], batch_size=64)
    global_model_state = base_server.state_dict()

    # 用于保存各个算法的测速结果
    results = {}

    # 通用参数封装
    client_kwargs = {
        "name": test_client_id, "dataset_id": config["system"]["dataset"],
        "epoch": config["client"]["num_local_epoch"], "model_name": config["system"]["model"],
        "lr": config["client"]["lr"], "batch_size": config["client"]["batch_size"],
        "momentum": config["client"]["momentum"]
    }

    # ==========================================
    # 1. 评估 FedAvg (Baseline)
    # ==========================================
    client_avg = FedClient(**client_kwargs)
    client_avg.load_trainset(client_data)
    client_avg.update(100, 1, global_model_state)
    results['FedAvg'] = profile_client(client_avg, "FedAvg", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 2. 评估 FedProx
    # ==========================================
    client_prox = FedProxClient(**client_kwargs)
    client_prox.load_trainset(client_data)
    client_prox.update(100, 1, global_model_state)
    results['FedProx'] = profile_client(client_prox, "FedProx", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 3. 评估 FedNova
    # ==========================================
    client_nova = FedNovaClient(**client_kwargs)
    client_nova.load_trainset(client_data)
    client_nova.update(100, 1, global_model_state)
    results['FedNova'] = profile_client(client_nova, "FedNova", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 4. 评估 FedDyn
    # ==========================================
    client_dyn = FedDynClient(**client_kwargs)
    client_dyn.load_trainset(client_data)
    client_dyn.update(100, 1, global_model_state)
    results['FedDyn'] = profile_client(client_dyn, "FedDyn", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 5. 评估 FedSAM (两次反向传播，开销应该最大)
    # ==========================================
    client_sam = FedSAMClient(**client_kwargs)
    client_sam.load_trainset(client_data)
    client_sam.update(100, 1, global_model_state)
    results['FedSAM'] = profile_client(client_sam, "FedSAM", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 6. 评估 SCAFFOLD
    # ==========================================
    server_scaf = ScaffoldServer([test_client_id], dataset_id=config["system"]["dataset"],
                                 model_name=config["system"]["model"], batch_size=64)
    client_scaf = ScaffoldClient(**client_kwargs)
    client_scaf.load_trainset(client_data)
    # SCAFFOLD 的 update 需要额外传入全局控制变量 scv_state
    client_scaf.update(100, 1, server_scaf.model.state_dict(), server_scaf.scv.state_dict())
    results['SCAFFOLD'] = profile_client(client_scaf, "SCAFFOLD", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 7. 评估 FedCM
    # ==========================================
    server_cm = FedCMServer([test_client_id], dataset_id=config["system"]["dataset"],
                            model_name=config["system"]["model"], batch_size=64)
    client_cm = FedCMClient(
        **client_kwargs, cm_alpha=config["client"].get("fedcm_alpha", 0.1)
    )
    client_cm.load_trainset(client_data)
    client_cm.update(100, 0, global_model_state, server_cm.direction_state_dict())
    cm_state, cm_n_data, cm_loss, cm_steps, cm_lr = client_cm.train()
    server_cm.selected_clients = [test_client_id]
    server_cm.rec(test_client_id, cm_state, cm_n_data, cm_loss, cm_steps, cm_lr)
    cm_model_state, _, _, cm_direction = server_cm.agg()
    client_cm.update(100, 1, cm_model_state, cm_direction)
    results['FedCM'] = profile_client(client_cm, "FedCM", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 8. 评估 FedSOL
    # ==========================================
    client_sol = FedSOLClient(
        **client_kwargs,
        rho=config["client"].get("fedsol_rho", 0.1),
        temperature=config["client"].get("fedsol_temperature", 3.0)
    )
    client_sol.load_trainset(client_data)
    client_sol.update(100, 1, global_model_state)
    results['FedSOL'] = profile_client(client_sol, "FedSOL", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 9. 评估 FedGeo (Ours)
    # ==========================================
    q_bits = config["system"].get("q_bits", 8)
    client_geo = GeoClient(**client_kwargs, beta=config["system"]["prox_beta"], q_bits=q_bits)
    client_geo.load_trainset(client_data)

    server_geo = GeoServer([test_client_id], dataset_id=config["system"]["dataset"],
                           model_name=config["system"]["model"], batch_size=64,
                           alpha=0.1, q_bits=q_bits)

    # FedGeo 需要先跑一次 Round 0 生成伪梯度，再测 Round 1 的真实开销
    global_payload = {"type": "full_model", "data": server_geo.state_dict(), "bn_stats": {}}
    client_geo.update(100, 0, global_payload)
    payload, n_data, loss = client_geo.train()
    server_geo.selected_clients = [test_client_id]
    server_geo.rec(test_client_id, payload, n_data, loss)
    q_grad_payload, _, _ = server_geo.agg()

    # 注入带有量化噪声和重建要求的 Payload
    client_geo.update(100, 1, q_grad_payload)
    results[f'FedGeo ({q_bits}-bit)'] = profile_client(client_geo, "FedGeo", args.runs)
    torch.cuda.empty_cache()

    # ==========================================
    # 输出对齐的表格结果
    # ==========================================
    print("\n" + "=" * 70)
    print(f"{'Algorithm':<18} | {'Wall-clock Time (ms)':<22} | {'Overhead vs FedAvg':<20}")
    print("-" * 70)

    base_time = results['FedAvg'][0]

    # 按耗时排序 (除 FedAvg 放第一位外)
    sorted_algos = ['FedAvg'] + sorted([k for k in results.keys() if k != 'FedAvg'], key=lambda k: results[k][0])

    for algo in sorted_algos:
        t_avg, t_std = results[algo]
        overhead = ((t_avg - base_time) / base_time) * 100

        if algo == 'FedAvg':
            overhead_str = "Baseline (0.00%)"
        else:
            overhead_str = f"+{overhead:.2f}%"

        print(f"{algo:<18} | {t_avg:>8.2f} ± {t_std:<9.2f} | {overhead_str:<20}")

    print("=" * 70)


if __name__ == "__main__":
    run_profiling()
