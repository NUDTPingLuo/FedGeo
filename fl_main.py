#!/usr/bin/env python
import os
import random
import json
import pickle
import argparse
import time
import yaml
from json import JSONEncoder
from tqdm import tqdm

from fed_baselines.client_base import FedClient
from fed_baselines.client_fedprox import FedProxClient
from fed_baselines.client_scaffold import ScaffoldClient
from fed_baselines.client_fednova import FedNovaClient
from fed_baselines.client_fedgeo import GeoClient
from fed_baselines.client_fedavgq import FedAvgQClient
from fed_baselines.client_fedcm import FedCMClient
from fed_baselines.client_fedsol import FedSOLClient
from fed_baselines.server_base import FedServer
from fed_baselines.server_scaffold import ScaffoldServer
from fed_baselines.server_fednova import FedNovaServer
from fed_baselines.server_fedgeo import GeoServer
from fed_baselines.server_fedavgq import FedAvgQServer
from fed_baselines.server_fedcm import FedCMServer

from postprocessing.recorder import Recorder
from preprocessing.baselines_dataloader import divide_data, divide_data_dirichlet, divide_data_dirichlet_resample
from utils.models import *
from utils.quantization import payload_nbytes, state_dict_nbytes

json_types = (list, dict, str, int, float, bool, type(None))

print("CUDA是否可用:", torch.cuda.is_available())
print("可用的GPU数量:", torch.cuda.device_count())
print("当前默认GPU设备:", torch.cuda.current_device() if torch.cuda.is_available() else "无GPU")
print("当前默认设备名称:",
      torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else "无GPU")


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


def format_number(value):
    """Use one canonical token (``1`` rather than both ``1`` and ``1.0``)."""
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def fed_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Yaml file for configuration')
    args = parser.parse_args()
    return args


def fed_run():
    args = fed_args()
    with open(args.config, "r") as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)

    algo_list = [
        "FedAvg", "Scaffold", "FedProx", "FedNova", "FedGeo",
        "FedCM", "FedSOL", "FedAvgQ"
    ]
    assert config["client"]["fed_algo"] in algo_list, "The federated learning algorithm is not supported"

    dataset_list = ['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100', 'ImageNet', 'LC25000']
    assert config["system"]["dataset"] in dataset_list, "The dataset is not supported"

    model_list = ["LeNet", 'CNN_MNIST', 'AlexCifarNet', "ResNet18", 'VGG11', "CNN", 'ResNet18_LC25000']
    assert config["system"]["model"] in model_list, "The model is not supported"

    np.random.seed(config["system"]["i_seed"])
    torch.manual_seed(config["system"]["i_seed"])
    random.seed(config["system"]["i_seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["system"]["i_seed"])

    client_dict = {}
    recorder = Recorder()
    recorder.res["config"] = config

    trainset_config, testset = divide_data_dirichlet(num_client=config["system"]["num_client"],
                                                     alpha=config["system"]["dirichlet_alpha"],
                                                     dataset_name=config["system"]["dataset"],
                                                     i_seed=config["system"]["i_seed"],
                                                     min_required=config["system"].get("min_client_samples", 256))
    max_acc = 0

    fed_algo = config["client"]["fed_algo"]
    common_client_args = {
        "dataset_id": config["system"]["dataset"],
        "epoch": config["client"]["num_local_epoch"],
        "model_name": config["system"]["model"],
        "lr": config["client"]["lr"],
        "batch_size": config["client"]["batch_size"],
        "momentum": config["client"]["momentum"],
    }

    for client_id in trainset_config['users']:
        if fed_algo == 'FedAvg':
            client = FedClient(client_id, **common_client_args)
        elif fed_algo == 'FedGeo':
            client = GeoClient(
                client_id, beta=config["system"]["prox_beta"],
                q_bits=config["system"].get("q_bits", 32), **common_client_args
            )
        elif fed_algo == 'FedAvgQ':
            client = FedAvgQClient(
                client_id, q_bits=config["system"].get("q_bits", 32),
                **common_client_args
            )
        elif fed_algo == 'FedCM':
            client = FedCMClient(
                client_id, cm_alpha=config["client"].get("fedcm_alpha", 0.1),
                **common_client_args
            )
        elif fed_algo == 'FedSOL':
            client = FedSOLClient(
                client_id,
                rho=config["client"].get("fedsol_rho", 0.1),
                temperature=config["client"].get("fedsol_temperature", 3.0),
                **common_client_args
            )
        elif fed_algo == 'Scaffold':
            client = ScaffoldClient(client_id, **common_client_args)
        elif fed_algo == 'FedProx':
            client = FedProxClient(client_id, **common_client_args)
        elif fed_algo == 'FedNova':
            client = FedNovaClient(client_id, **common_client_args)

        client.load_trainset(trainset_config['user_data'][client_id])
        client_dict[client_id] = client

    common_server_args = {
        "dataset_id": config["system"]["dataset"],
        "model_name": config["system"]["model"],
        "batch_size": config["client"]["batch_size"],
    }
    log_run_tag = "{}_{}_alpha{}_beta{}_seed{}_bit{}".format(
        fed_algo, config["system"]["dataset"],
        format_number(config["system"]["dirichlet_alpha"]),
        format_number(config["system"]["prox_beta"]),
        config["system"]["i_seed"], config["system"].get("q_bits", 32)
    )
    if fed_algo in ('FedAvg', 'FedProx', 'FedSOL'):
        fed_server = FedServer(trainset_config['users'], **common_server_args)
    elif fed_algo == 'FedGeo':
        fed_server = GeoServer(trainset_config['users'], dataset_id=config["system"]["dataset"],
                               model_name=config["system"]["model"],
                               batch_size=config["client"]["batch_size"],
                               alpha=config["system"]["dirichlet_alpha"],
                               q_bits=config["system"].get("q_bits", 32),
                               run_tag=log_run_tag)
    elif fed_algo == 'FedAvgQ':
        fed_server = FedAvgQServer(
            trainset_config['users'], alpha=config["system"]["dirichlet_alpha"],
            q_bits=config["system"].get("q_bits", 32), run_tag=log_run_tag,
            **common_server_args
        )
    elif fed_algo == 'FedCM':
        fed_server = FedCMServer(trainset_config['users'], **common_server_args)
    elif fed_algo == 'Scaffold':
        fed_server = ScaffoldServer(trainset_config['users'], **common_server_args)
        scv_state = fed_server.scv.state_dict()
    elif fed_algo == 'FedNova':
        fed_server = FedNovaServer(trainset_config['users'], **common_server_args)

    fed_server.load_testset(testset)
    global_state_dict = fed_server.state_dict()

    incremental_algorithms = ('FedGeo', 'FedAvgQ')
    if fed_algo in incremental_algorithms:
        global_payload = {"type": "full_model", "data": global_state_dict, "bn_stats": {}}
        global_payload["wire_bytes"] = payload_nbytes(global_payload)

    pbar = tqdm(range(config["system"]["num_round"]))
    total_round = config["system"]["num_round"]

    # 🚀 定义每轮参与的客户端比例 (默认 0.5)
    client_fraction = float(config["client"].get("fraction", 0.5))
    if not 0.0 < client_fraction <= 1.0:
        raise ValueError("client.fraction must be in (0, 1]")
    num_select = max(1, int(len(trainset_config['users']) * client_fraction))
    recorder.res['server'].update({
        'uplink_bytes': [], 'downlink_bytes': [], 'communication_bytes': [],
        'cumulative_communication_bytes': [], 'round_time_seconds': []
    })
    cumulative_bytes = 0

    for global_round in pbar:
        round_start = time.perf_counter()
        # 🚀 1. 每一轮开始时，随机抽取指定数量的客户端
        selected_clients = random.sample(trainset_config['users'], num_select)
        fed_server.selected_clients = selected_clients

        # 🚀 2. 全网广播阶段 (Broadcast Phase)
        # 对于下发压缩增量(q_grad)的算法，所有客户端必须同步更新状态，防止 NoneType
        if fed_algo in incremental_algorithms:
            payload_bytes = (
                global_payload["wire_bytes"] if "wire_bytes" in global_payload
                else payload_nbytes(global_payload)
            )
            round_downlink_bytes = int(payload_bytes) * len(trainset_config['users'])
            for client_id in trainset_config['users']:
                client_dict[client_id].update(total_round, global_round, global_payload)
        elif fed_algo == 'FedCM':
            round_downlink_bytes = (
                state_dict_nbytes(global_state_dict) +
                state_dict_nbytes(fed_server.direction_state_dict())
            ) * len(selected_clients)
        else:
            round_downlink_bytes = state_dict_nbytes(global_state_dict) * len(selected_clients)

        # 🚀 3. 本地训练阶段 (Local Training Phase)
        # 只有被选中的客户端才进行反向传播训练，真正节省算力！
        for client_id in selected_clients:
            if fed_algo == 'FedAvg':
                client_dict[client_id].update(total_round, global_round, global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

            elif fed_algo in incremental_algorithms:
                payload, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, payload, n_data, loss)

            elif fed_algo == 'FedCM':
                client_dict[client_id].update(
                    total_round, global_round, global_state_dict,
                    fed_server.direction_state_dict()
                )
                state_dict, n_data, loss, local_steps, current_lr = client_dict[client_id].train()
                fed_server.rec(
                    client_dict[client_id].name, state_dict, n_data, loss,
                    local_steps, current_lr
                )

            elif fed_algo == 'FedSOL':
                client_dict[client_id].update(total_round, global_round, global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

            elif fed_algo == 'Scaffold':
                client_dict[client_id].update(total_round, global_round, global_state_dict, scv_state)
                state_dict, n_data, loss, delta_ccv_state = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, delta_ccv_state)

            elif fed_algo == 'FedProx':
                client_dict[client_id].update(total_round, global_round, global_state_dict)
                state_dict, n_data, loss = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss)

            elif fed_algo == 'FedNova':
                client_dict[client_id].update(total_round, global_round, global_state_dict)
                state_dict, n_data, loss, coeff, norm_grad = client_dict[client_id].train()
                fed_server.rec(client_dict[client_id].name, state_dict, n_data, loss, coeff, norm_grad)

        # 🚀 4. 聚合阶段 (移除 fed_server.select_clients())
        if fed_algo in ('FedAvg', 'FedProx', 'FedSOL'):
            global_state_dict, avg_loss, _ = fed_server.agg()
        elif fed_algo in incremental_algorithms:
            global_payload, avg_loss, _ = fed_server.agg()
            global_state_dict = fed_server.state_dict()
        elif fed_algo == 'FedCM':
            global_state_dict, avg_loss, _, _ = fed_server.agg()
        elif fed_algo == 'Scaffold':
            global_state_dict, avg_loss, _, scv_state = fed_server.agg()
        elif fed_algo == 'FedNova':
            global_state_dict, avg_loss, _ = fed_server.agg()

        accuracy = fed_server.test()
        round_uplink_bytes = int(getattr(fed_server, "last_uplink_bytes", 0))
        round_communication = round_uplink_bytes + round_downlink_bytes
        cumulative_bytes += round_communication

        recorder.res['server']['iid_accuracy'].append(accuracy)
        recorder.res['server']['train_loss'].append(avg_loss)
        recorder.res['server']['uplink_bytes'].append(round_uplink_bytes)
        recorder.res['server']['downlink_bytes'].append(round_downlink_bytes)
        recorder.res['server']['communication_bytes'].append(round_communication)
        recorder.res['server']['cumulative_communication_bytes'].append(cumulative_bytes)
        recorder.res['server']['round_time_seconds'].append(time.perf_counter() - round_start)
        fed_server.flush()

        if max_acc < accuracy:
            max_acc = accuracy
        pbar.set_description(
            f'Global Round: {global_round} | Train loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | Max Acc: {max_acc:.4f}')

        if not os.path.exists(config["system"]["res_root"]):
            os.makedirs(config["system"]["res_root"])

        dataset = config["system"]["dataset"]
        lr = config["client"]["lr"]
        alpha = config["system"]["dirichlet_alpha"]
        beta = config["system"]["prox_beta"]
        random_seed = config["system"]["i_seed"]

        result_tags = [
            fed_algo, dataset, 'alpha{}'.format(format_number(alpha)),
            'lr{}'.format(format_number(lr)), 'beta{}'.format(format_number(beta)),
            'seed{}'.format(random_seed)
        ]
        q_bits = int(config["system"].get("q_bits", 32))
        if fed_algo in incremental_algorithms and q_bits != 32:
            result_tags.append('bit{}'.format(q_bits))
        if fed_algo == 'FedCM':
            result_tags.append('cmalpha{}'.format(format_number(config["client"].get("fedcm_alpha", 0.1))))
        if fed_algo == 'FedSOL':
            result_tags.append('rho{}'.format(format_number(config["client"].get("fedsol_rho", 0.1))))
        file_name = "[{}]".format(",".join("'{}'".format(tag) for tag in result_tags))
        # save_dir = os.path.join(config["system"]["res_root"])

        save_dir = os.path.join(config["system"]["res_root"],
                                dataset,
                                "alpha" + str(alpha),
                                "lr" + str(lr),
                                "beta" + str(beta),
                                # "bit" + str(q_bits),
                                fed_algo
                                )

        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "w") as jsfile:
            json.dump(recorder.res, jsfile, cls=PythonObjectEncoder)


if __name__ == "__main__":
    fed_run()
