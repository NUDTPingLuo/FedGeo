import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import os
import PIL
import random
import numpy as np
from torch.utils.data import Subset
from torchvision.transforms import InterpolationMode
from torchvision import datasets, transforms
from torch.utils.data import random_split


def load_data(name, root='./data', download=True, save_pre_data=True):

    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN', "ImageNet", 'CIFAR100', 'LC25000', 'GastroVision']
    assert name in data_dict, "The dataset is not present"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split= 'letters', download=download, transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split= 'letters', download=download, transform=transform)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)

    elif name == 'CIFAR10':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'CIFAR100':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
        trainset.targets = torch.Tensor(trainset.labels)
        testset.targets = torch.Tensor(testset.labels)


    elif name == 'ImageNet':  # 指代 Tiny ImageNet
        # 1. 定义标准的 ImageNet 均值和标准差 (Tiny ImageNet 完全适用)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            # 2. 核心增强：四周填充 4 像素后，随机裁剪回 64x64
            transforms.RandomCrop(64, padding=4),
            # 3. 核心增强：50% 概率水平翻转
            transforms.RandomHorizontalFlip(),
            # (可选) 保留你原有的轻微颜色抖动
            transforms.ColorJitter(hue=.05, saturation=.05),
            # 之前你的 RandomRotation(20) 容易引入黑边，对于 64x64 的小图建议去掉，保留 Crop 和 Flip 足够了
            transforms.ToTensor(),
            # 4. 取消注释：执行标准化操作
            normalize,
        ])

        test_transform = transforms.Compose([
            # 5. 致命修复：测试集绝对不能有任何随机增强（删除了 ColorJitter）
            transforms.ToTensor(),
            normalize,
        ])

        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)

        # 保持你原有的 target 转换逻辑
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    # === LC25000 数据集 ===
    elif name == 'LC25000':
        # --- 加速版训练增强 ---
        train_val_transform = transforms.Compose([
            transforms.Resize((128, 128)),  # 从 112 → 128，有利于 GPU 速度
            transforms.RandomHorizontalFlip(p=0.5),  # 保留最有效且最便宜的增强
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # --- 测试集 ---
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        trainset = torchvision.datasets.ImageFolder(
            root='./data/LC25000/train',
            transform=train_val_transform
        )
        testset = torchvision.datasets.ImageFolder(
            root='./data/LC25000/test',
            transform=test_transform
        )

    # === GastroVision 数据集 ===
    elif name == 'GastroVision':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # 加载完整数据集（未划分）
        full_dataset = datasets.ImageFolder(root='./data/GastroVision', transform=transform)

        # 按 80% / 20% 划分为训练集和测试集
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        trainset, testset = random_split(full_dataset, [train_size, test_size])

        # 记录类别数
        trainset.targets = torch.Tensor([y for _, y in trainset.dataset.samples])
        testset.targets = torch.Tensor([y for _, y in testset.dataset.samples])


    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'ImageNet': 200,
        'CIFAR100': 100,
        'LC25000': 5,  # LC25000 数据集共 5 类
        'GastroVision': 27  # GastroVision 数据集共 27 类
    }

    len_classes = len_classes_dict[name]

    return trainset, testset, len_classes


def divide_data(num_client=1, num_local_class=10, dataset_name='emnist', i_seed=0):

    torch.manual_seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)

    num_classes = len_classes
    if num_local_class == -1:
        num_local_class = num_classes
    assert 0 < num_local_class <= num_classes, "number of local class should smaller than global number of class"

    trainset_config = {'users': [],
                       'user_data': {},
                       'num_samples': []}
    config_division = {}  # Count of the classes for division
    config_class = {}  # Configuration of class distribution in clients
    config_data = {}  # Configuration of data indexes for each class : Config_data[cls] = [0, []] | pointer and indexes

    for i in range(num_client):
        config_class['f_{0:05d}'.format(i)] = []
        for j in range(num_local_class):
            cls = (i+j) % num_classes
            if cls not in config_division:
                config_division[cls] = 1
                config_data[cls] = [0, []]

            else:
                config_division[cls] += 1
            config_class['f_{0:05d}'.format(i)].append(cls)

    # print(config_class)
    # print(config_division)

    for cls in config_division.keys():
        indexes = torch.nonzero(trainset.targets == cls)
        num_datapoint = indexes.shape[0]
        indexes = indexes[torch.randperm(num_datapoint)]
        num_partition = num_datapoint // config_division[cls]
        for i_partition in range(config_division[cls]):
            if i_partition == config_division[cls] - 1:
                config_data[cls][1].append(indexes[i_partition * num_partition:])
            else:
                config_data[cls][1].append(indexes[i_partition * num_partition: (i_partition + 1) * num_partition])

    for user in tqdm(config_class.keys()):
        user_data_indexes = torch.tensor([])
        for cls in config_class[user]:
            user_data_index = config_data[cls][1][config_data[cls][0]]
            user_data_indexes = torch.cat((user_data_indexes, user_data_index))
            config_data[cls][0] += 1
        user_data_indexes = user_data_indexes.squeeze().int().tolist()
        user_data = Subset(trainset, user_data_indexes)
        #user_targets = trainset.target[user_data_indexes.tolist()]
        trainset_config['users'].append(user)
        trainset_config['user_data'][user] = user_data
        trainset_config['num_samples'] = len(user_data)

    #
    # test_loader = DataLoader(trainset_config['user_data']['f_00001'])
    # for i, (x,y) in enumerate(test_loader):
    #     print(i)
    #     print(y)


    return trainset_config, testset

def divide_data_dirichlet(num_client=10, alpha=0.1, dataset_name='emnist', i_seed=0, min_required=256):
    """
    优化版 Dirichlet 数据划分函数：
    - 不使用 while True，直接一次性划分；
    - 若客户端样本不足，则自动从样本多的客户端“借”；
    - 保留 Dirichlet 分布的非IID特性；
    - 打印每个客户端的样本数和类别数。
    """

    torch.manual_seed(i_seed)
    np.random.seed(i_seed)

    # === 加载数据集 ===
    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)
    num_classes = len_classes

    # === 初始化 ===
    trainset_config = {'users': [], 'user_data': {}, 'num_samples': []}
    client_indices = {i: [] for i in range(num_client)}

    # === Step 1: 按 Dirichlet 分布划分每个类别 ===
    for cls in range(num_classes):
        cls_indices = np.where(np.array(trainset.targets) == cls)[0]
        np.random.shuffle(cls_indices)
        proportions = np.random.dirichlet(alpha=np.ones(num_client) * alpha)
        expected_counts = proportions / proportions.sum() * len(cls_indices)
        counts = np.floor(expected_counts).astype(int)
        # Largest-remainder allocation keeps every sample while introducing
        # less bias than assigning all rounding leftovers to low client IDs.
        remainder = len(cls_indices) - counts.sum()
        if remainder > 0:
            residual_order = np.argsort(-(expected_counts - counts))
            counts[residual_order[:remainder]] += 1
        proportions = counts

        start = 0
        for client_id, count in enumerate(proportions):
            if count > 0:
                client_indices[client_id].extend(cls_indices[start:start + count])
                start += count

    # === Step 2: 修正样本量不足的客户端 ===
    client_sizes = [len(client_indices[i]) for i in range(num_client)]
    total_needed = {i: min_required - size for i, size in enumerate(client_sizes) if size < min_required}

    if total_needed:
        print(f"[Adjusting clients with < {min_required} samples...]")
        for poor_id, need in total_needed.items():
            for rich_id, rich_size in sorted(enumerate(client_sizes), key=lambda x: -x[1]):
                if rich_id == poor_id or rich_size <= min_required:
                    continue
                take = min(need, rich_size - min_required)
                if take > 0:
                    move_indices = client_indices[rich_id][:take]
                    client_indices[poor_id].extend(move_indices)
                    client_indices[rich_id] = client_indices[rich_id][take:]
                    client_sizes[poor_id] += take
                    client_sizes[rich_id] -= take
                    need -= take
                    if need <= 0:
                        break

    # === Step 3: 封装划分结果 ===
    for client_id in range(num_client):
        indices = client_indices[client_id]
        user_id = f'f_{client_id:05d}'
        client_subset = Subset(trainset, indices)
        trainset_config['users'].append(user_id)
        trainset_config['user_data'][user_id] = client_subset
        trainset_config['num_samples'].append(len(indices))

        # 统计每个客户端的标签种类数
        labels = [int(trainset.targets[i]) for i in indices]
        num_unique_labels = len(set(labels))
        print(f'Client {user_id} | #Samples: {len(indices)} | #Classes: {num_unique_labels}')

    return trainset_config, testset


def divide_data_dirichlet_resample(num_client=50, alpha=0.01, dataset_name='MNIST', i_seed=0, min_required=1):
    """
    带“安全逃生舱”的 Dirichlet 划分法 (Safe Resampling Dirichlet):
    - 前 50 次尝试纯净的 while 循环重采样。
    - 如果检测到极端 Non-IID 导致的数学死局，自动打破死循环。
    - 启动微量补偿机制，仅为缺少数据的客户端补足最小要求，极限保留分布纯度。
    """
    torch.manual_seed(i_seed)
    np.random.seed(i_seed)

    trainset, testset, len_classes = load_data(dataset_name, download=True, save_pre_data=False)
    num_classes = len_classes

    if isinstance(trainset.targets, torch.Tensor):
        targets = trainset.targets.numpy()
    else:
        targets = np.array(trainset.targets)

    min_size = 0
    max_retries = 50  # 🚀 设置最大重试次数，防止死循环
    retries = 0

    # === 核心 1：尝试纯净重采样 ===
    while min_size < min_required and retries < max_retries:
        client_indices = [[] for _ in range(num_client)]

        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)

            proportions = np.random.dirichlet(np.repeat(alpha, num_client))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            client_indices_k = np.split(idx_k, proportions)
            for i in range(num_client):
                client_indices[i] += client_indices_k[i].tolist()

        min_size = min([len(idx_j) for idx_j in client_indices])
        retries += 1

    # === 核心 2：安全逃生舱 (微量借用补偿) ===
    if min_size < min_required:
        print(f"⚠️ 警告: 触发极端 Non-IID 数学死局。已启动微量补偿机制补足 min_required={min_required}...")
        client_sizes = [len(client_indices[i]) for i in range(num_client)]

        for poor_id, size in enumerate(client_sizes):
            while len(client_indices[poor_id]) < min_required:
                # 找到当前数据最充裕的客户端
                rich_id = np.argmax([len(idx) for idx in client_indices])
                # 从“大户”那里借 1 个样本给“贫困户”
                borrowed_sample = client_indices[rich_id].pop()
                client_indices[poor_id].append(borrowed_sample)

    # === 封装划分结果 ===
    trainset_config = {'users': [], 'user_data': {}, 'num_samples': []}

    for client_id in range(num_client):
        indices = client_indices[client_id]
        np.random.shuffle(indices)

        user_id = f'f_{client_id:05d}'
        client_subset = Subset(trainset, indices)

        trainset_config['users'].append(user_id)
        trainset_config['user_data'][user_id] = client_subset
        trainset_config['num_samples'].append(len(indices))

        labels = [int(targets[i]) for i in indices]
        num_unique_labels = len(set(labels))
        print(f'Client {user_id} | #Samples: {len(indices)} | #Classes: {num_unique_labels}')

    return trainset_config, testset

if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN', 'LC25000', 'GastroVision']

    for name in data_dict:
        print(name)
        divide_data(num_client=20, num_local_class=2, dataset_name=name, i_seed=0)
