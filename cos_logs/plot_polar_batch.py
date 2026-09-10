import argparse
import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import rcParams


def plot_polar_trajectory_grouped(csv_path, save_path=None):
    """
    绘制极坐标收敛轨迹图 (强制极简文本动态聚类版)。
    自动从文件名中提取 Dataset 和 alpha 用于生成图表标题。
    """
    # ===== 1. 设置顶级学术图表字体与排版样式 =====
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'DejaVu Sans'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 找不到文件: {csv_path}，请确保它与本脚本在同一目录下！")

    # 💡 利用正则表达式从文件名中提取数据集和 alpha
    filename = os.path.basename(csv_path)
    match = re.search(r"(.*?)_alpha([0-9\.]+)", filename)

    if match:
        dataset_name = match.group(1)
        alpha_val = match.group(2)

        # 💡 核心修改 1：如果文件名不包含 FedAvg，说明是 FedGeo 的数据，在标题前强制加上 FedGeo
        if "FedAvg" not in filename:
            # 防止原本文件名里已经有 FedGeo 导致重复 (例如 FedGeo_CIFAR10)
            display_name = f"FedGeo_{dataset_name}" if "FedGeo" not in dataset_name else dataset_name
        else:
            display_name = dataset_name

        title_text = f"{display_name} ($\\alpha$={alpha_val})"

        if save_path is None:
            # 保持原文件名规则不变，防止破坏 LaTeX 里面的图片引用路径
            save_path = f"cos_polar_{dataset_name}_alpha{alpha_val}.pdf"
    else:
        title_text = "Client Gradient Direction Trajectory"
        if save_path is None:
            safe_name = filename.replace('.csv', '.pdf')
            save_path = f"cos_polar_{safe_name}"

    # ===== 加载数据 =====
    df = pd.read_csv(csv_path)
    rounds = df['Round'].values
    r = rounds / rounds.max()  # 半径归一化
    client_cols = [c for c in df.columns if c != 'Round']

    # 建立强制极简映射，无视乱码和长字符串
    client_mapping = {c: str(i) for i, c in enumerate(client_cols)}

    # ===== 2. 预处理：计算每个 Client 的最终收敛角度 =====
    final_angles_rad = {}
    all_theta_deg = []

    for c in client_cols:
        cos_vals = np.clip(df[c].values, -1.0, 1.0)
        theta_rad = np.arccos(cos_vals)
        all_theta_deg.extend(np.degrees(theta_rad))
        # 取最后 3 轮的角度平均值作为该客户端的最终位置
        final_angles_rad[c] = np.mean(theta_rad[-3:])

    # ===== 3. 核心算法：按角度对 Client 进行动态聚类 =====
    sorted_clients = sorted(client_cols, key=lambda x: final_angles_rad[x])

    groups = []
    threshold_deg = 3.0  # 聚类容差
    threshold_rad = np.radians(threshold_deg)

    if sorted_clients:
        current_group = [sorted_clients[0]]
        for c in sorted_clients[1:]:
            group_avg_rad = np.mean([final_angles_rad[x] for x in current_group])
            if abs(final_angles_rad[c] - group_avg_rad) <= threshold_rad:
                current_group.append(c)
            else:
                groups.append(current_group)
                current_group = [c]
        groups.append(current_group)

    # ===== 4. 初始化画布并画线 =====
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='polar')

    for c in client_cols:
        cos_vals = np.clip(df[c].values, -1.0, 1.0)
        theta_rad = np.arccos(cos_vals)

        points = np.array([theta_rad, r]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(r.min(), r.max())
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(r)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)

    # ===== 5. 绘制合并后的极简智能标签 =====
    for group in groups:
        group_center_rad = np.mean([final_angles_rad[c] for c in group])
        nums = [client_mapping[c] for c in group]
        label_text = f"C{','.join(nums)}"

        ax.text(group_center_rad, r[-1] + 0.08, label_text,
                ha='center', va='center', fontsize=11, weight='bold')

    # ===== 6. 统一扇形视野与刻度 =====
    min_deg = 0.0
    max_deg = 120.0  # 固定最大角度为 120 度

    ax.set_thetamin(min_deg)
    ax.set_thetamax(max_deg)

    # 让扇形完美居中对称 (中间的角度垂直朝上)
    mid_angle_deg = (min_deg + max_deg) / 2
    ax.set_theta_offset(np.radians(90 - mid_angle_deg))

    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # 动态生成真实轮次标签
    max_r = rounds.max()
    labels = [f"{int(max_r * v)}" for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    ax.set_yticklabels(labels, size=10, alpha=0.8)

    # 强制固定刻度间隔为 20 度
    step = 20.0
    xticks_deg = np.arange(min_deg, max_deg + 0.1, step)
    ax.set_xticks(np.radians(xticks_deg))
    ax.set_xticklabels([f"{d:.1f}" for d in xticks_deg], size=11, weight='bold')

    ax.grid(True, linestyle='-', linewidth=0.8, alpha=0.5, color='gray')
    ax.spines['polar'].set_linewidth(1.2)

    # 💡 核心修改 2：把 y 参数从 1.12 调小到 1.06，拉近标题和图表的距离
    plt.title(title_text, va='bottom', y=0.85, fontsize=16, weight='bold')

    # ===== 7. 保存 =====
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 成功保存: {save_path}")


if __name__ == "__main__":
    # ===== 配置命令行参数解析器 =====
    parser = argparse.ArgumentParser(description="Batch Plot Polar Trajectory for FedGeo")

    parser.add_argument("--csv", type=str, default=None,
                        help="指定要画图的 CSV 文件名")

    parser.add_argument("--batch", action="store_true",
                        help="自动寻找当前目录下所有的 *_cos_matrix.csv 并批量画图")

    args = parser.parse_args()

    DEFAULT_CSV = "CIFAR10_alpha0.01_cos_matrix.csv"

    try:
        if args.batch:
            csv_files = glob.glob("*_cos_matrix.csv")
            if not csv_files:
                print("⚠️ 当前目录下没有找到任何以 '_cos_matrix.csv' 结尾的文件！")
            else:
                print(f"🔍 开启批量模式，共找到 {len(csv_files)} 个文件，开始绘图...")
                for file in csv_files:
                    print(f"\n⏳ 正在处理: {file}")
                    plot_polar_trajectory_grouped(csv_path=file)
                print("\n🎉 全部文件批量绘图完成！")

        else:
            target_file = args.csv if args.csv else DEFAULT_CSV
            print(f"⏳ 正在处理单个文件: {target_file}")
            plot_polar_trajectory_grouped(csv_path=target_file)
            print("\n🎉 绘图完成！")

    except Exception as e:
        print(f"❌ 运行出错: {e}")
