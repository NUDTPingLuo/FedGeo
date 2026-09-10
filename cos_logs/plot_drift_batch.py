import argparse
import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import defaultdict


def plot_drift_cancellation_combined(dataset_name, csv_files, beta=0.5, save_dir=None, output_filename=None):
    """
    Plot empirical dispersion D_t and reference-steering magnitude S_t.
    D_t = B_sq; S_t = 2 * beta * Cancel_Base. These are not cancelling terms.
    """
    # ===== 1. 设置顶级学术图表字体与排版样式 =====
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # 预设高对比度鲜明颜色 (拒绝暗淡，提升视觉冲击力)
    alpha_colors = {
        0.01: '#E41A1C',  # 鲜红色
        0.1: '#377EB8',  # 鲜蓝色
        1.0: '#4DAF4A'  # 翠绿色
    }
    fallback_palette = plt.get_cmap('tab10').colors

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    max_val = 0.0

    # ===== 2. 预处理并排序文件 (按 alpha 从小到大) =====
    parsed_files = []
    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        match = re.search(r".*?_?(.*?)_alpha([0-9\.]+)_drift_cancellation", filename)
        if match:
            alpha_val = float(match.group(2))
            parsed_files.append((alpha_val, csv_path))

    parsed_files.sort(key=lambda x: x[0])

    color_idx = 0
    # ===== 3. 开始遍历绘图 =====
    for alpha_val, csv_path in parsed_files:
        df = pd.read_csv(csv_path)
        if 'Round' not in df.columns or 'B_sq' not in df.columns or 'Cancel_Base' not in df.columns:
            continue

        rounds = df['Round'].values
        b_sq = df['B_sq'].values
        cancellation = df['Cancel_Base'].values * (2.0 * beta)

        current_max = max(np.max(b_sq), np.max(cancellation))
        if current_max > max_val:
            max_val = current_max

        color = alpha_colors.get(alpha_val, fallback_palette[color_idx % 10])
        color_idx += 1

        alpha_label = "1" if alpha_val == 1.0 else str(alpha_val)

        # 💡 极简图例，移除复杂公式，保留核心标识
        ax.plot(rounds, b_sq, label=rf'$D_t$ ($\alpha={alpha_label}$)',
                color=color, linewidth=2.5, linestyle='--')

        ax.plot(rounds, cancellation, label=rf'$S_t$ ($\alpha={alpha_label}$)',
                color=color, linewidth=2.5, linestyle='-')

        # 💡 已移除原有的 fill_between (阴影代码)

    # ===== 4. 图像细节设置 =====
    if max_val > 0:
        ax.set_ylim(0, max_val * 1.1)

    ax.set_xlabel('Communication Rounds', size=16)
    ax.set_ylabel('Magnitude', size=16)
    ax.tick_params(axis='both', labelsize=14)

    # 图例设置：精简紧凑，放置在右上角
    ax.legend(prop={'size': 14}, loc='upper right', ncol=2, framealpha=0.95, columnspacing=1.0)

    ax.grid(alpha=0.3, linestyle=':')  # 改为点状网格线，进一步减少对实线和虚线的视觉干扰

    display_name = {'ImageNet': 'Tiny-ImageNet', 'CIFAR10': 'CIFAR-10',
                    'CIFAR100': 'CIFAR-100', 'FashionMNIST': 'Fashion-MNIST'}.get(dataset_name, dataset_name)
    plt.suptitle(display_name, size=20, fontweight='bold')

    # ===== 5. 保存逻辑 =====
    if save_dir is None:
        save_dir = os.path.join('..', 'plot', dataset_name, 'theory')
    os.makedirs(save_dir, exist_ok=True)

    save_name = output_filename or f"FedGeo_{dataset_name}_drift_cancel_combined_clean.pdf"
    save_path = os.path.join(save_dir, save_name)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 成功保存(极简高对比版): {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Plot Drift Cancellation for FedGeo (Clean Version)")
    parser.add_argument("--batch", action="store_true", help="自动寻找所有的 *_drift_cancellation.csv 并合并")
    parser.add_argument("--beta", type=float, default=0.5, help="算法的 beta 值 (默认: 0.5)")
    args = parser.parse_args()

    try:
        csv_files = glob.glob("*_drift_cancellation.csv")
        if not csv_files:
            print("⚠️ 未找到任何 '_drift_cancellation.csv' 结尾的文件！")
        else:
            grouped_files = defaultdict(list)
            for f in csv_files:
                match = re.search(r".*?_?(.*?)_alpha([0-9\.]+)_drift_cancellation", os.path.basename(f))
                if match:
                    grouped_files[match.group(1).replace("FedGeo_", "")].append(f)

            for dataset_name, files in grouped_files.items():
                print(f"📊 正在处理数据集: {dataset_name}")
                plot_drift_cancellation_combined(dataset_name, files, beta=args.beta)

            print("🎉 所有合并绘图任务完成！")
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
