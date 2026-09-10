import subprocess
import yaml
import sys

config_path = "config/test_config.yaml"

# ==========================================
# 🚀 自动化映射配置
# ==========================================
# 数据集与模型的自动化映射字典
dataset_model_map = {
    "MNIST": "LeNet",
    "FashionMNIST": "LeNet",
    "CIFAR10": "CNN",
    "CIFAR100": "CNN",
    "ImageNet": "ResNet18"
}

# 你想要跑的数据集列表（可随时增删）
datasets = ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100", "ImageNet"]

# 5 个随机种子
seeds = [0]

# 算法列表
algorithms = [
    # "FedAvg",
    "FedGeo",
    # "FedProx",
    # "FedNova",
    # "Scaffold",
]

# 统一的异构度参数
alphas = [0.01, 0.1, 1]

# ==========================================
# FedGeo 专属参数
# ==========================================
q_bits_list = [32]
betas = [0.5]  # 🔥 新增：FedGeo专属的 Beta 列表

# ==========================================
# 动态计算总实验次数
# ==========================================
total_experiments = 0
for algo in algorithms:
    if algo == "FedGeo":
        # 乘以 beta 的长度
        total_experiments += len(datasets) * len(seeds) * len(alphas) * len(q_bits_list) * len(betas)
    else:
        total_experiments += len(datasets) * len(seeds) * len(alphas)

experiment_count = 0

# ==========================================
# 自动化执行流
# ==========================================
for dataset in datasets:
    model_name = dataset_model_map[dataset]  # 自动匹配对应模型

    for s in seeds:
        for algo in algorithms:
            if algo == "FedGeo":
                # FedGeo 需要遍历 alpha, q_bits 和 beta
                for alpha in alphas:
                    for qb in q_bits_list:
                        for beta in betas:  # 🔥 新增：Beta 循环
                            experiment_count += 1

                            with open(config_path, "r") as f:
                                config = yaml.safe_load(f)

                            # ✅ 严格根据你真实的 test_config.yaml 结构写入
                            if "system" not in config: config["system"] = {}
                            if "client" not in config: config["client"] = {}

                            # 写入 dataset 和 model (都在 system 节点下)
                            config["system"]["dataset"] = dataset
                            config["system"]["model"] = model_name

                            # 写入系统和算法配置
                            config["system"]["i_seed"] = s
                            config["system"]["dirichlet_alpha"] = alpha
                            config["system"]["q_bits"] = qb
                            config["system"]["prox_beta"] = beta  # 🔥 写入当前 Beta

                            config["client"]["fed_algo"] = algo

                            with open(config_path, "w") as f:
                                yaml.safe_dump(config, f, default_flow_style=False)

                            print(
                                f"\n[{experiment_count}/{total_experiments}] ===== Dataset: {dataset} ({model_name}) | Algo: FedGeo | Seed: {s} | Alpha: {alpha} | Q-Bits: {qb} | Beta: {beta} =====")

                            subprocess.run([sys.executable, "fl_main.py", "--config", config_path], check=True)

            else:
                # 其他 Baseline 只需要遍历 alpha
                for alpha in alphas:
                    experiment_count += 1

                    with open(config_path, "r") as f:
                        config = yaml.safe_load(f)

                    # ✅ 严格根据你真实的 test_config.yaml 结构写入
                    if "system" not in config: config["system"] = {}
                    if "client" not in config: config["client"] = {}

                    # 写入 dataset 和 model
                    config["system"]["dataset"] = dataset
                    config["system"]["model"] = model_name

                    # 写入系统和算法配置
                    config["system"]["i_seed"] = s
                    config["system"]["dirichlet_alpha"] = alpha
                    config["client"]["fed_algo"] = algo

                    # Baseline 强行设为全量传输 (32-bit)，并随便给个默认 beta 防止报错
                    config["system"]["q_bits"] = 32
                    if "prox_beta" not in config["system"]:
                        config["system"]["prox_beta"] = 1.0

                    with open(config_path, "w") as f:
                        yaml.safe_dump(config, f, default_flow_style=False)

                    print(
                        f"\n[{experiment_count}/{total_experiments}] ===== Dataset: {dataset} ({model_name}) | Algo: {algo} | Seed: {s} | Alpha: {alpha} =====")

                    subprocess.run([sys.executable, "fl_main.py", "--config", config_path], check=True)
