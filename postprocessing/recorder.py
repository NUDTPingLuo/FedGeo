import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import json
from json import JSONEncoder
import pickle
import re
from matplotlib import rcParams
import seaborn as sns
import os
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import csv
import pandas as pd

json_types = (list, dict, str, int, float, bool, type(None))
DEFAULT_DATASETS = ('MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100', 'ImageNet')
DATASET_ALIASES = {'IMAGENET': 'ImageNet'}
DEFAULT_PLOT_SEEDS = (0, 1, 42, 999, 2026)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DEFAULT_RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'results')
DEFAULT_PLOT_ROOT = os.path.join(PROJECT_ROOT, 'plot')
FEDGEO_GREEN = '#2ca02c'
FEDGEO_GREEN_SHADES = {
    0.01: '#006d2c',
    0.1: '#31a354',
    1.0: '#74c476',
}


def _format_number(value):
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def parse_experiment_label(label):
    """Parse result names without depending on token order.

    Older FedGeo files place ``bit`` before ``seed`` while newer files place it
    after ``seed``.  Prefix-based parsing supports both layouts.
    """
    tokens = re.findall(r"'([^']+)'", os.path.basename(label))
    if len(tokens) < 2:
        return None

    def find_token(prefix, cast, default=None):
        for token in tokens[2:]:
            if token.startswith(prefix):
                try:
                    return cast(token[len(prefix):])
                except (TypeError, ValueError):
                    return default
        return default

    alpha = find_token('alpha', float)
    lr = find_token('lr', float)
    beta = find_token('beta', float)
    seed = find_token('seed', int)
    bit = find_token('bit', int, 32)
    if alpha is None or lr is None or beta is None or seed is None:
        return None

    known_prefixes = ('alpha', 'lr', 'beta', 'seed', 'bit')
    extras = tuple(sorted(
        token for token in tokens[2:]
        if not token.startswith(known_prefixes)
    ))
    raw_alpha = next(
        (token for token in tokens[2:] if token.startswith('alpha')), ''
    )
    return {
        'algorithm': tokens[0],
        'dataset': DATASET_ALIASES.get(tokens[1], tokens[1]),
        'alpha': alpha,
        'lr': lr,
        'beta': beta,
        'seed': seed,
        'bit': bit,
        'extras': extras,
        'raw_alpha': raw_alpha,
        'tokens': tokens,
    }


class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, json_types):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}


def as_python_object(dct):
    if '_python_object' in dct:
        return pickle.loads(dct['_python_object'].encode('latin-1'))
    return dct


class Recorder(object):
    def __init__(self):
        self.res_list = []
        self.records = []
        self.res = {'server': {'iid_accuracy': [], 'train_loss': []},
                    'clients': {'iid_accuracy': [], 'train_loss': []}}

    def load(self, filename, label=None):
        """
        Load the result files
        :param filename: Name of the result file
        :param label: Label for the result file
        """
        label = os.path.basename(filename) if label is None else label
        with open(filename) as json_file:
            res = json.load(json_file, object_hook=as_python_object)
        self.res_list.append((res, label))
        self.records.append({
            'res': res,
            'label': label,
            'path': os.path.abspath(filename),
            'meta': parse_experiment_label(label),
        })

    def load_dataset_results(self, results_root=None, datasets=DEFAULT_DATASETS,
                             seeds=DEFAULT_PLOT_SEEDS, clear=True):
        """Recursively load result files from ``results/<dataset>`` folders.

        The legacy ``results/test`` directory is intentionally ignored.  Files
        are deduplicated by experiment configuration and seed.  When both
        ``alpha1`` and ``alpha1.0`` names exist, the canonical ``alpha1`` name
        is preferred.
        """
        results_root = DEFAULT_RESULTS_ROOT if results_root is None else os.path.abspath(results_root)
        if clear:
            self.res_list = []
            self.records = []

        selected = {}
        allowed_seeds = set(int(seed) for seed in seeds) if seeds is not None else None
        for dataset in datasets:
            dataset_dir = os.path.join(results_root, dataset)
            if not os.path.isdir(dataset_dir):
                print('Warning: result directory not found:', dataset_dir)
                continue
            for current_root, _, filenames in os.walk(dataset_dir):
                for filename in filenames:
                    meta = parse_experiment_label(filename)
                    if meta is None or meta['dataset'] != dataset:
                        continue
                    if allowed_seeds is not None and meta['seed'] not in allowed_seeds:
                        continue
                    key = (
                        meta['algorithm'], meta['dataset'], meta['alpha'], meta['lr'],
                        meta['beta'], meta['seed'], meta['bit'], meta['extras']
                    )
                    canonical_alpha = 'alpha{}'.format(_format_number(meta['alpha']))
                    score = (
                        int(meta['raw_alpha'] == canonical_alpha),
                        os.path.getmtime(os.path.join(current_root, filename))
                    )
                    candidate = (score, os.path.join(current_root, filename), filename)
                    if key not in selected or candidate[0] > selected[key][0]:
                        selected[key] = candidate

        for _, filename, label in sorted(selected.values(), key=lambda item: item[1]):
            try:
                self.load(filename, label)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print('Warning: failed to load {}: {}'.format(filename, exc))

        print('Loaded {} result files from dataset folders (seeds={}).'.format(
            len(self.records), tuple(seeds) if seeds is not None else 'all'
        ))
        return len(self.records)

    def _iter_records(self, seeds=DEFAULT_PLOT_SEEDS):
        allowed_seeds = set(int(seed) for seed in seeds) if seeds is not None else None
        for record in self.records:
            meta = record['meta']
            if meta is None:
                continue
            if allowed_seeds is not None and meta['seed'] not in allowed_seeds:
                continue
            yield record['res'], meta

    @staticmethod
    def _plot_accuracy_series(ax, arrays, label, color, linestyle='-', linewidth=2.2):
        if not arrays:
            return False
        min_len = min(len(array) for array in arrays)
        if min_len == 0:
            return False
        values = np.array([np.asarray(array[:min_len], dtype=float) for array in arrays])
        mean = values.mean(axis=0)
        lower = values.min(axis=0)
        upper = values.max(axis=0)
        ax.plot(mean, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth)
        ax.fill_between(np.arange(min_len), lower, upper, color=color, alpha=0.12)
        return True

    @staticmethod
    def _tighten_accuracy_axis(ax, warmup_fraction=0.1, tick=0.05,
                               minimum_lower_bound=0.05):
        """Fit the y-axis to the post-warmup accuracy range of an ablation.

        Both mean curves and their range bands are considered, so the tighter
        limits reveal differences without clipping seed variability.  Bounds
        are rounded to readable accuracy increments and never start at zero.
        """
        line_x = [np.asarray(line.get_xdata(), dtype=float) for line in ax.get_lines()]
        finite_x = [values[np.isfinite(values)] for values in line_x if len(values)]
        if not finite_x:
            return
        maximum_round = max(values.max() for values in finite_x if len(values))
        warmup_round = maximum_round * float(warmup_fraction)

        y_values = []
        for line in ax.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x >= warmup_round)
            if mask.any():
                y_values.append(y[mask])
        for collection in ax.collections:
            for path in collection.get_paths():
                vertices = np.asarray(path.vertices, dtype=float)
                if vertices.ndim != 2 or vertices.shape[1] < 2:
                    continue
                mask = (
                    np.isfinite(vertices[:, 0]) & np.isfinite(vertices[:, 1]) &
                    (vertices[:, 0] >= warmup_round)
                )
                if mask.any():
                    y_values.append(vertices[mask, 1])
        if not y_values:
            return

        data_minimum = min(values.min() for values in y_values)
        data_maximum = max(values.max() for values in y_values)
        data_span = max(data_maximum - data_minimum, tick)
        padding = max(0.04 * data_span, 0.01)
        lower = max(
            minimum_lower_bound,
            np.floor((data_minimum - padding) / tick) * tick
        )
        upper = min(1.0, np.ceil((data_maximum + padding) / tick) * tick)
        if upper - lower < 0.15:
            lower = max(minimum_lower_bound, upper - 0.15)
        ax.set_ylim(float(lower), float(upper))

    @staticmethod
    def _finish_figure(fig, ax, title, save_path, legend_columns=1):
        ax.set_xlabel('Communication Rounds', size=13)
        ax.set_ylabel('Testing Accuracy', size=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10, ncol=legend_columns, framealpha=0.95)
        ax.set_title(title, fontsize=16, fontweight='bold')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print('Saved:', save_path)

    def _plot_legacy(self, figsize=(6, 5)):
        """
        Plot testing accuracy (mean ± range) across different seeds with custom figure size and Nature-style color scheme.
        """

        # ===== 设置字体嵌入 =====
        global Dirichlet
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'

        # 设置 seaborn 风格和调色板
        sns.set_style("white")
        color_palette = sns.color_palette("colorblind")

        # ===== 1. 颜色映射 (Color Map) =====
        # 充分利用 colorblind 调色板，确保 7 种颜色对比极其鲜明
        base_color_map = {
            'FedAvg': color_palette[7],  # 灰色 (作为经典的 Baseline 色调)
            'Scaffold': color_palette[0],  # 深蓝色
            'FedProx': color_palette[9],  # 青色/天蓝色
            'FedNova': color_palette[1],  # 橙色
            'FedDyn': color_palette[4],  # 紫色
            'FedSAM': color_palette[5],  # 棕色
            'FedCM': color_palette[6],
            'FedSOL': color_palette[3],
            'FedAvgQ': color_palette[8],
            'FedGeo': FEDGEO_GREEN,  # 🟢 绿色实线突出主要贡献
        }

        # ===== 2. 线型映射 (Linestyle Map) =====
        # 标准线型不够用，引入高级自定义线型 (offset, (on_off_sequence))
        line_style_map = {
            'FedAvg': '--',  # Baseline 使用虚线
            'Scaffold': '--',  # 虚线 (Dashed)
            'FedProx': '-.',  # 点划线 (Dash-dot)
            'FedNova': ':',  # 点线 (Dotted)
            'FedDyn': (0, (3, 1, 1, 1)),  # 密集的单点划线 (Densely dash-dotted)
            'FedSAM': (0, (5, 2, 5, 2, 1, 2)),  # 双点长划线 (Dash-dash-dot)
            'FedCM': (0, (4, 2)),
            'FedSOL': (0, (3, 1, 1, 1)),
            'FedAvgQ': '--',
            'FedGeo': '-',  # 实线 (Ours，靠颜色和线宽区分)
        }

        # ===== 按算法 + 数据集 + Dirichlet 聚合同类实验（不同 seed） =====
        grouped_results = defaultdict(list)
        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            Algorithm = matches[0]
            Dataset = matches[1]
            Dirichlet = matches[2]
            lr = matches[3]
            beta = matches[4]
            key = (Algorithm, Dataset, Dirichlet)
            grouped_results[key].append(np.array(res['server']['iid_accuracy']))

        # ===== 开始绘图 =====
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            # 对齐长度（防止不同实验长度略有差异）
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            # 计算均值、最大值、最小值
            mean_acc = acc_array.mean(axis=0)
            max_acc = acc_array.max(axis=0)
            min_acc = acc_array.min(axis=0)

            # 绘图属性
            color = base_color_map.get(Algorithm, 'black')
            linestyle = line_style_map.get(Algorithm, '-')

            # ✅ 修复：将之前的 ECGR 统一改为 FedGeo，凸显自己的算法
            alpha_value = 0.4 if "FedGeo" in Algorithm else 0.15
            linewidth_value = 3.0 if "FedGeo" in Algorithm else 1.5

            # 绘制平均线
            ax.plot(
                mean_acc,
                label=Algorithm,
                alpha=1.0,
                linewidth=linewidth_value,
                color=color,
                linestyle=linestyle
            )

            # 绘制阴影（最大值与最小值之间区域）
            ax.fill_between(
                range(min_len),
                min_acc,
                max_acc,
                color=color,
                alpha=alpha_value
            )

        # 创建 inset axes（放大图）
        inset_ax = inset_axes(
            ax,
            width="30%",  # inset 宽度
            height="30%",  # inset 高度
            bbox_to_anchor=(-0.05, -0.65, 1, 1),  # 右侧中间
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        start_i, end_i = 90, 100

        for (Algorithm, Dataset, Dirichlet), acc_list in grouped_results.items():
            # 对齐长度
            min_len = min(len(a) for a in acc_list)
            acc_array = np.array([a[:min_len] for a in acc_list])

            # 计算均值（只取 90-100 范围）
            mean_acc = acc_array.mean(axis=0)[start_i:end_i]

            color = base_color_map.get(Algorithm, 'black')
            linestyle = line_style_map.get(Algorithm, '-')

            # ✅ 放大图中同样保持 Ours 的加粗特征
            linewidth_value = 3.0 if "FedGeo" in Algorithm else 1.5

            x_range = np.arange(start_i, end_i)

            # ⭐ 只画曲线，不画阴影
            inset_ax.plot(
                x_range,
                mean_acc,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth_value,
            )

        # inset 图细节
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(alpha=0.3)

        # ===== 在主图 ↔ inset 之间添加连接线（Nature风格）=====
        try:
            ax.indicate_inset_zoom(inset_ax, edgecolor="black", linewidth=1.2)
        except Exception as e:
            print("⚠️ inset zoom connection unavailable:", e)

        # ===== 图像细节设置 =====
        ax.set_xlabel('Epochs', size=12)
        ax.set_ylabel('Testing Accuracy', size=12)
        # 微调 Legend 大小，防止 7 个算法把图表遮挡太多
        # ax.legend(prop={'size': 11}, loc='upper left', framealpha=0.9)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle(Dataset, size=16, fontweight='bold')

        # ===== 保存结果 =====
        save_dir = os.path.join('..', 'plot', Dataset)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{Dataset}_{Dirichlet}_{lr}_{beta}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    def plot(self, figsize=(6, 5), seeds=DEFAULT_PLOT_SEEDS, output_root=None):
        """Plot the main lr=0.01 comparison, including FedCM and FedSOL.

        One figure is generated for each dataset/alpha pair.  FedGeo uses
        beta=0.5; quantized runs and LR-ablation runs are excluded.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        algorithms = [
            'FedAvg', 'Scaffold', 'FedProx', 'FedNova', 'FedDyn',
            'FedSAM', 'FedCM', 'FedSOL', 'FedGeo'
        ]
        palette = sns.color_palette('colorblind', n_colors=10)
        baseline_color_indices = (0, 1, 3, 4, 5, 6, 7, 9)
        colors = {
            algorithm: palette[color_index]
            for algorithm, color_index in zip(algorithms[:-1], baseline_color_indices)
        }
        colors['FedGeo'] = FEDGEO_GREEN
        linestyles = {
            'FedAvg': '--', 'Scaffold': '-.', 'FedProx': ':',
            'FedNova': (0, (5, 2)), 'FedDyn': (0, (3, 1, 1, 1)),
            'FedSAM': (0, (5, 2, 1, 2)), 'FedCM': (0, (4, 1)),
            'FedSOL': (0, (2, 1)), 'FedGeo': '-'
        }
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            algorithm = meta['algorithm']
            if algorithm not in algorithms or meta['bit'] != 32:
                continue
            if not np.isclose(meta['lr'], 0.01):
                continue
            if algorithm == 'FedGeo' and not np.isclose(meta['beta'], 0.5):
                continue
            grouped[(meta['dataset'], meta['alpha'], algorithm)].append(
                np.asarray(res['server']['iid_accuracy'])
            )

        saved_paths = []
        dataset_alpha_pairs = sorted(
            set((key[0], key[1]) for key in grouped),
            key=lambda item: (DEFAULT_DATASETS.index(item[0]) if item[0] in DEFAULT_DATASETS else 99,
                              item[1])
        )
        for dataset, alpha in dataset_alpha_pairs:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            plotted = False
            for algorithm in algorithms:
                arrays = grouped.get((dataset, alpha, algorithm), [])
                plotted |= self._plot_accuracy_series(
                    ax, arrays, algorithm, colors[algorithm],
                    linestyles[algorithm], 3.0 if algorithm == 'FedGeo' else 1.8
                )
            if not plotted:
                plt.close(fig)
                continue
            alpha_text = _format_number(alpha)
            save_path = os.path.join(
                output_root, dataset, 'main_comparison',
                '{}_alpha{}_lr0.01.pdf'.format(dataset, alpha_text)
            )
            self._finish_figure(
                fig, ax, '{} ($\\alpha$={})'.format(dataset, alpha_text),
                save_path, legend_columns=2
            )
            saved_paths.append(save_path)
        return saved_paths

    def _print_table_metrics_legacy(self):
        """
        自动计算并打印符合 LaTeX 表格格式的统计指标：
        - Acc: 百分制 (如 87.24 ± 3.81)
        - Rounds: 达到同分布下 FedAvg 平均最高准确率所需的轮次。若未达到，记为 N/A。
        - Speedup: FedAvg轮次 / 当前算法轮次。若未达到，记为 N/A。
        """
        # ===== 1. 按算法、数据集、分布聚合同类实验 =====
        grouped_results = defaultdict(list)
        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            if len(matches) >= 3:
                Algorithm = matches[0]
                Dataset = matches[1]
                Dirichlet = matches[2]
                grouped_results[(Algorithm, Dataset, Dirichlet)].append(np.array(res['server']['iid_accuracy']))

        # ===== 2. 第一遍扫描：计算所有方法的 Mean Acc 和 Std Acc (转为百分制) =====
        stats = {}
        for key, acc_list in grouped_results.items():
            max_accs = []
            for acc_array in acc_list:
                if len(acc_array) > 0:
                    max_accs.append(np.max(acc_array) * 100.0)

            stats[key] = {
                'mean_acc': np.mean(max_accs) if max_accs else 0.0,
                'std_acc': np.std(max_accs) if max_accs else 0.0
            }

        # ===== 3. 第二遍扫描：计算 Rounds 并严格判定 N/A =====
        for key, acc_list in grouped_results.items():
            Algorithm, Dataset, Dirichlet = key

            # 寻找同数据集、同分布的 FedAvg 作为靶心
            baseline_key = next(
                (k for k in stats.keys() if k[0].lower() == 'fedavg' and k[1] == Dataset and k[2] == Dirichlet), None)

            if baseline_key:
                target_acc = stats[baseline_key]['mean_acc'] / 100.0
            else:
                target_acc = stats[key]['mean_acc'] / 100.0

            # 核心拦截逻辑：如果该算法的平均最高准确率，连 FedAvg 的线都没摸到 (容差 1e-4)，直接判死刑 N/A
            if key != baseline_key and (stats[key]['mean_acc'] / 100.0) < (target_acc - 1e-4):
                stats[key]['rounds'] = 'N/A'
            else:
                reached_rounds = []
                for acc_array in acc_list:
                    if len(acc_array) == 0:
                        continue

                    indices = np.where(acc_array >= (target_acc - 1e-6))[0]
                    if len(indices) > 0:
                        reached_rounds.append(indices[0] + 1)
                    else:
                        # 对于达标的算法，如果个别 seed 运气不好没达到，惩罚为其最大跑动轮次
                        reached_rounds.append(len(acc_array))

                stats[key]['rounds'] = round(np.mean(reached_rounds)) if reached_rounds else 0

        # ===== 4. 计算加速比并打印格式化表格 =====
        print("=" * 85)
        print(
            f"{'Dataset':<15} | {'Dirichlet':<10} | {'Algorithm':<12} | {'Acc ± Std (%)':<16} | {'Rounds':<8} | {'Speedup'}")
        print("-" * 85)

        sorted_keys = sorted(stats.keys(), key=lambda x: (x[1], x[2], x[0]))

        for key in sorted_keys:
            Algorithm, Dataset, Dirichlet = key
            current_stats = stats[key]

            baseline_key = next(
                (k for k in stats.keys() if k[0].lower() == 'fedavg' and k[1] == Dataset and k[2] == Dirichlet), None)

            # 格式化输出
            acc_str = f"{current_stats['mean_acc']:.2f} ± {current_stats['std_acc']:.2f}"

            # 只有在 rounds 不是 N/A 的情况下，才计算 Speedup
            rounds_val = current_stats['rounds']
            if rounds_val == 'N/A':
                rounds_str = "N/A"
                speedup_str = "N/A"
            else:
                rounds_str = f"{rounds_val}"
                if baseline_key and baseline_key in stats and stats[baseline_key]['rounds'] != 'N/A' and rounds_val > 0:
                    baseline_rounds = stats[baseline_key]['rounds']
                    speedup = baseline_rounds / rounds_val
                    speedup_str = f"{speedup:.2f}x"
                else:
                    speedup_str = "1.00x"

            print(
                f"{Dataset:<15} | {Dirichlet:<10} | {Algorithm:<12} | {acc_str:<16} | {rounds_str:<8} | {speedup_str}")
        print("=" * 85)

    def print_table_metrics(self, seeds=DEFAULT_PLOT_SEEDS):
        """Report integer rounds-to-target on the cross-seed mean curve.

        The shared target is the peak of the FedAvg mean curve, NOT the mean
        of per-run peaks. Missing or incomplete runs raise an error. Failure
        to cross by round 100 is censored (>100), never counted as a hit.
        Accuracy retains the existing per-run peak mean/population SD.
        """
        algorithms = {
            'FedAvg', 'Scaffold', 'FedProx', 'FedNova', 'FedDyn',
            'FedSAM', 'FedCM', 'FedSOL', 'FedGeo'
        }
        grouped = defaultdict(dict)
        for res, meta in self._iter_records(seeds):
            algorithm = meta['algorithm']
            if algorithm not in algorithms or meta['bit'] != 32:
                continue
            if not np.isclose(meta['lr'], 0.01):
                continue
            if algorithm == 'FedGeo' and not np.isclose(meta['beta'], 0.5):
                continue
            key = (algorithm, meta['dataset'], meta['alpha'])
            if meta['seed'] in grouped[key]:
                raise ValueError('Duplicate main-comparison run: {} seed {}'.format(key, meta['seed']))
            grouped[key][meta['seed']] = np.asarray(res['server']['iid_accuracy'], dtype=float)

        stats = {}
        mean_curves = {}
        for key, runs in grouped.items():
            if set(runs) != set(seeds):
                raise ValueError('Incomplete seeds for {}: {}'.format(key, sorted(runs)))
            arrays = [runs[seed] for seed in seeds]
            if any(len(array) != 100 or not np.isfinite(array).all() for array in arrays):
                raise ValueError('Expected 100 finite post-round accuracies for {}'.format(key))
            mean_curves[key] = np.mean(arrays, axis=0)
            maxima = [array.max() * 100.0 for array in arrays]
            stats[key] = {
                'mean_acc': float(np.mean(maxima)),
                'std_acc': float(np.std(maxima)),
            }

        for key, mean_curve in mean_curves.items():
            algorithm, dataset, alpha = key
            baseline_key = ('FedAvg', dataset, alpha)
            target = float(mean_curves[baseline_key].max())
            indices = np.flatnonzero(mean_curve >= target)
            stats[key]['target_acc'] = 100.0 * target
            stats[key]['rounds'] = int(indices[0] + 1) if len(indices) else None
        for key, values in stats.items():
            baseline = stats[('FedAvg', key[1], key[2])]
            values['speedup'] = baseline['rounds'] / values['rounds'] if values['rounds'] else None

        print('=' * 88)
        print("{:<15} | {:<8} | {:<12} | {:<18} | {:<8} | {}".format(
            'Dataset', 'Alpha', 'Algorithm', 'Acc ± Std (%)', 'Rounds', 'Speedup'
        ))
        print('-' * 88)
        for key in sorted(stats, key=lambda item: (item[1], item[2], item[0])):
            algorithm, dataset, alpha = key
            values = stats[key]
            baseline = stats.get(('FedAvg', dataset, alpha))
            rounds = values['rounds']
            if rounds is None:
                rounds_text, speedup_text = '>100', 'N/A'
            else:
                rounds_text = str(rounds)
                if baseline and baseline.get('rounds') is not None and rounds > 0:
                    speedup_text = '{:.2f}x'.format(baseline['rounds'] / rounds)
                else:
                    speedup_text = '1.00x'
            print("{:<15} | {:<8} | {:<12} | {:<18} | {:<8} | {}".format(
                dataset, _format_number(alpha), algorithm,
                '{:.2f} ± {:.2f}'.format(values['mean_acc'], values['std_acc']),
                rounds_text, speedup_text
            ))
        print('=' * 88)
        return stats

    def plot_fedgeo_beta0_vs_fedavg(self, figsize=(6, 5),
                                    seeds=DEFAULT_PLOT_SEEDS, output_root=None):
        """Compare FedGeo beta=0 with FedAvg at lr=0.01.

        A single figure is produced per dataset; all three alpha values share
        that figure.  Color represents alpha and line style represents method.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] not in ('FedAvg', 'FedGeo'):
                continue
            if meta['bit'] != 32 or not np.isclose(meta['lr'], 0.01):
                continue
            if meta['algorithm'] == 'FedGeo' and not np.isclose(meta['beta'], 0.0):
                continue
            grouped[(meta['dataset'], meta['alpha'], meta['algorithm'])].append(
                np.asarray(res['server']['iid_accuracy'], dtype=float)
            )

        baseline_colors = {0.01: '#d62728', 0.1: '#1f77b4', 1.0: '#9467bd'}
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            plotted = False
            for alpha in (0.01, 0.1, 1.0):
                for algorithm, linestyle in (('FedAvg', '--'), ('FedGeo', '-')):
                    color = (
                        FEDGEO_GREEN_SHADES[alpha]
                        if algorithm == 'FedGeo' else baseline_colors[alpha]
                    )
                    label = '{} ($\\alpha$={})'.format(
                        'FedGeo ($\\beta$=0)' if algorithm == 'FedGeo' else 'FedAvg',
                        _format_number(alpha)
                    )
                    plotted |= self._plot_accuracy_series(
                        ax, grouped.get((dataset, alpha, algorithm), []),
                        label, color, linestyle,
                        2.8 if algorithm == 'FedGeo' else 2.0
                    )
            if not plotted:
                plt.close(fig)
                continue
            save_path = os.path.join(
                output_root, dataset, 'supplementary',
                '{}_FedGeo_beta0_vs_FedAvg_lr0.01.pdf'.format(dataset)
            )
            self._tighten_accuracy_axis(ax)
            self._finish_figure(
                fig, ax, '{}: FedGeo $\\beta$=0 vs FedAvg'.format(dataset),
                save_path, legend_columns=2
            )
            saved_paths.append(save_path)
        return saved_paths

    def plot_lr_ablation_fedavg_fedgeo(self, figsize=(6, 5),
                                       seeds=DEFAULT_PLOT_SEEDS,
                                       learning_rates=(0.1, 0.01, 0.001),
                                       output_root=None):
        """Plot FedAvg/FedGeo(beta=0.5) learning-rate ablations.

        Three figures are produced per dataset, one for each alpha.  Every
        figure contains both algorithms at all requested learning rates.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        learning_rates = tuple(float(value) for value in learning_rates)
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] not in ('FedAvg', 'FedGeo') or meta['bit'] != 32:
                continue
            if not any(np.isclose(meta['lr'], value) for value in learning_rates):
                continue
            if meta['algorithm'] == 'FedGeo' and not np.isclose(meta['beta'], 0.5):
                continue
            lr_key = next(value for value in learning_rates if np.isclose(meta['lr'], value))
            grouped[(meta['dataset'], meta['alpha'], meta['algorithm'], lr_key)].append(
                np.asarray(res['server']['iid_accuracy'], dtype=float)
            )

        lr_colors = {0.1: '#d62728', 0.01: '#1f77b4', 0.001: '#2ca02c'}
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            for alpha in (0.01, 0.1, 1.0):
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                plotted = False
                for lr in learning_rates:
                    color = lr_colors.get(lr, 'black')
                    for algorithm, linestyle in (('FedAvg', '--'), ('FedGeo', '-')):
                        label = '{} (lr={})'.format(algorithm, _format_number(lr))
                        plotted |= self._plot_accuracy_series(
                            ax, grouped.get((dataset, alpha, algorithm, lr), []),
                            label, color, linestyle,
                            2.7 if algorithm == 'FedGeo' else 1.9
                        )
                if not plotted:
                    plt.close(fig)
                    continue
                alpha_text = _format_number(alpha)
                save_path = os.path.join(
                    output_root, dataset, 'supplementary', 'lr_ablation',
                    '{}_alpha{}_FedAvg_vs_FedGeo_beta0.5_lr.pdf'.format(
                        dataset, alpha_text
                    )
                )
                self._tighten_accuracy_axis(ax)
                self._finish_figure(
                    fig, ax, '{} ($\\alpha$={})'.format(dataset, alpha_text),
                    save_path, legend_columns=2
                )
                saved_paths.append(save_path)
        return saved_paths

    def plot_high_lr_beta_ablation(self, figsize=(6, 5),
                                   seeds=DEFAULT_PLOT_SEEDS,
                                   output_root=None):
        """Compare FedGeo beta=0.1/0.5 and FedAvg at the fixed lr=0.1.

        One figure is generated for every dataset/alpha pair.  At lr=0.1,
        FedGeo beta=0.1 is the highlighted green solid curve; beta=0.5 and
        FedAvg are comparison curves and therefore use distinct dashed styles.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] not in ('FedAvg', 'FedGeo'):
                continue
            if meta['bit'] != 32 or not np.isclose(meta['lr'], 0.1):
                continue
            if meta['algorithm'] == 'FedGeo':
                if not any(np.isclose(meta['beta'], beta) for beta in (0.1, 0.5)):
                    continue
            elif not np.isclose(meta['beta'], 0.0):
                continue
            grouped[(
                meta['dataset'], meta['alpha'], meta['algorithm'], meta['beta']
            )].append(np.asarray(res['server']['iid_accuracy'], dtype=float))

        series = (
            ('FedGeo', 0.1, 'FedGeo ($\\beta$=0.1)', FEDGEO_GREEN, '-', 3.0),
            ('FedGeo', 0.5, 'FedGeo ($\\beta$=0.5)', '#1f77b4', '--', 2.2),
            ('FedAvg', 0.0, 'FedAvg', '#7f7f7f', (0, (5, 2, 1, 2)), 2.0),
        )
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            for alpha in (0.01, 0.1, 1.0):
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                plotted = False
                for algorithm, beta, label, color, linestyle, linewidth in series:
                    plotted |= self._plot_accuracy_series(
                        ax, grouped.get((dataset, alpha, algorithm, beta), []),
                        label, color, linestyle, linewidth
                    )
                if not plotted:
                    plt.close(fig)
                    continue

                alpha_text = _format_number(alpha)
                save_path = os.path.join(
                    output_root, dataset, 'supplementary',
                    'high_lr_beta_ablation',
                    '{}_alpha{}_FedGeo_beta0.1_beta0.5_vs_FedAvg_lr0.1.pdf'.format(
                        dataset, alpha_text
                    )
                )
                self._tighten_accuracy_axis(ax)
                self._finish_figure(
                    fig, ax,
                    '{} ($\\alpha$={}, lr=0.1)'.format(dataset, alpha_text),
                    save_path
                )
                saved_paths.append(save_path)
        return saved_paths

    def plot_quantization_ablation_fedavgq_fedgeo(
            self, figsize=(6, 5), seeds=DEFAULT_PLOT_SEEDS,
            bits=(2, 4, 8), output_root=None):
        """Compare FedAvg-Q and FedGeo(beta=0.5) at matched bit widths.

        The learning rate is fixed at 0.01.  Three figures are produced per
        dataset, one for each alpha, with 2/4/8-bit curves in each figure.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        bits = tuple(int(value) for value in bits)
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] not in ('FedAvgQ', 'FedGeo'):
                continue
            if meta['bit'] not in bits or not np.isclose(meta['lr'], 0.01):
                continue
            if meta['algorithm'] == 'FedGeo' and not np.isclose(meta['beta'], 0.5):
                continue
            grouped[(meta['dataset'], meta['alpha'], meta['algorithm'], meta['bit'])].append(
                np.asarray(res['server']['iid_accuracy'], dtype=float)
            )

        bit_colors = {2: '#d62728', 4: '#2ca02c', 8: '#1f77b4'}
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            for alpha in (0.01, 0.1, 1.0):
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                plotted = False
                for bit in bits:
                    color = bit_colors.get(bit, 'black')
                    for algorithm, linestyle in (('FedAvgQ', '--'), ('FedGeo', '-')):
                        display_name = 'FedAvg-Q' if algorithm == 'FedAvgQ' else 'FedGeo'
                        plotted |= self._plot_accuracy_series(
                            ax, grouped.get((dataset, alpha, algorithm, bit), []),
                            '{} ({}-bit)'.format(display_name, bit),
                            color, linestyle, 2.7 if algorithm == 'FedGeo' else 1.9
                        )
                if not plotted:
                    plt.close(fig)
                    continue
                alpha_text = _format_number(alpha)
                save_path = os.path.join(
                    output_root, dataset, 'supplementary', 'quantization_ablation',
                    '{}_alpha{}_FedAvgQ_vs_FedGeo_beta0.5_bits.pdf'.format(
                        dataset, alpha_text
                    )
                )
                self._tighten_accuracy_axis(ax)
                self._finish_figure(
                    fig, ax, '{} ($\\alpha$={}, lr=0.01)'.format(dataset, alpha_text),
                    save_path, legend_columns=2
                )
                saved_paths.append(save_path)
        return saved_paths

    def plot_beta_ablation(self, target_algo='FedGeo', figsize=(6, 5),
                           seeds=DEFAULT_PLOT_SEEDS,
                           betas=(0.1, 0.5, 1.0), output_root=None):
        """Plot the lr=0.01, full-precision alignment-strength ablation.

        Only the requested beta values are loaded.  The default beta=0.5
        configuration is highlighted with the green solid curve; the other
        settings are dashed controls.
        """
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        betas = tuple(float(value) for value in betas)
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] != target_algo or meta['bit'] != 32:
                continue
            if not np.isclose(meta['lr'], 0.01):
                continue
            if not any(np.isclose(meta['beta'], value) for value in betas):
                continue
            beta_key = next(value for value in betas if np.isclose(meta['beta'], value))
            grouped[(meta['dataset'], meta['alpha'], beta_key)].append(
                np.asarray(res['server']['iid_accuracy'], dtype=float)
            )

        styles = {
            0.1: ('#1f77b4', '--', 2.2),
            0.5: (FEDGEO_GREEN, '-', 3.0),
            1.0: ('#ff7f0e', ':', 2.2),
        }
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            for alpha in (0.01, 0.1, 1.0):
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                plotted = False
                for beta in betas:
                    color, linestyle, linewidth = styles.get(
                        beta, ('#7f7f7f', '-.', 2.2)
                    )
                    plotted |= self._plot_accuracy_series(
                        ax, grouped.get((dataset, alpha, beta), []),
                        '{} ($\\beta$={})'.format(target_algo, _format_number(beta)),
                        color, linestyle, linewidth
                    )
                if not plotted:
                    plt.close(fig)
                    continue
                alpha_text = _format_number(alpha)
                save_path = os.path.join(
                    output_root, dataset, 'ablation',
                    '{}_{}_alpha{}_beta_ablation.pdf'.format(
                        target_algo, dataset, alpha_text
                    )
                )
                self._tighten_accuracy_axis(ax)
                self._finish_figure(
                    fig, ax, '{} ($\\alpha$={}, lr=0.01)'.format(dataset, alpha_text),
                    save_path
                )
                saved_paths.append(save_path)
        return saved_paths

    def plot_bit_ablation(self, target_algo='FedGeo', figsize=(6, 5),
                          seeds=DEFAULT_PLOT_SEEDS, bits=(2, 4, 8, 32),
                          output_root=None):
        """Plot the within-FedGeo bit-width ablation at lr=0.01, beta=0.5."""
        output_root = DEFAULT_PLOT_ROOT if output_root is None else output_root
        bits = tuple(int(value) for value in bits)
        grouped = defaultdict(list)
        for res, meta in self._iter_records(seeds):
            if meta['algorithm'] != target_algo or meta['bit'] not in bits:
                continue
            if not np.isclose(meta['lr'], 0.01) or not np.isclose(meta['beta'], 0.5):
                continue
            grouped[(meta['dataset'], meta['alpha'], meta['bit'])].append(
                np.asarray(res['server']['iid_accuracy'], dtype=float)
            )

        styles = {
            2: ('#d62728', ':', 2.2),
            4: ('#ff7f0e', '-.', 2.2),
            8: ('#1f77b4', '--', 2.2),
            32: (FEDGEO_GREEN, '-', 3.0),
        }
        saved_paths = []
        for dataset in DEFAULT_DATASETS:
            for alpha in (0.01, 0.1, 1.0):
                fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
                plotted = False
                for bit in bits:
                    color, linestyle, linewidth = styles.get(
                        bit, ('#7f7f7f', (0, (3, 1, 1, 1)), 2.2)
                    )
                    label = '{}-bit{}'.format(bit, ' (Full)' if bit == 32 else '')
                    plotted |= self._plot_accuracy_series(
                        ax, grouped.get((dataset, alpha, bit), []),
                        label, color, linestyle, linewidth
                    )
                if not plotted:
                    plt.close(fig)
                    continue
                alpha_text = _format_number(alpha)
                save_path = os.path.join(
                    output_root, dataset, 'ablation',
                    '{}_{}_alpha{}_bit_ablation.pdf'.format(
                        target_algo, dataset, alpha_text
                    )
                )
                self._tighten_accuracy_axis(ax)
                self._finish_figure(
                    fig, ax, '{} ($\\alpha$={}, lr=0.01)'.format(dataset, alpha_text),
                    save_path
                )
                saved_paths.append(save_path)
        return saved_paths

    def plot_multi_client(self, figsize=(6, 5)):
        """
        绘制 FedGeo 与 FedAvg 在 50 客户端、50% 随机采样场景下的鲁棒性对比图。
        相同 alpha 使用相同颜色；FedGeo 为实线，FedAvg 为虚线。
        风格沿用极致紧凑排版与学术级大字号。
        """
        # ===== 1. 设置字体和样式 =====
        rcParams['pdf.fonttype'] = 42
        rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = 'DejaVu Sans'
        sns.set_style("white")

        # FedGeo 使用绿色实线；FedAvg 使用不同颜色的虚线。
        baseline_alpha_colors = {
            0.01: '#d62728',
            0.1: '#1f77b4',
            1.0: '#9467bd',
        }

        # 线型映射：FedGeo 实线展现稳定性，FedAvg 虚线对比漂移
        algo_linestyles = {
            'FedGeo': '-',
            'FedAvg': '--'
        }

        # ===== 2. 聚合同类实验 =====
        grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for res, label in self.res_list:
            matches = re.findall(r"'([^']+)'", label)
            if len(matches) < 3:
                continue

            Algorithm = matches[0]
            if Algorithm not in ['FedGeo', 'FedAvg']:
                continue

            Dataset = matches[1]

            # 统一处理 alpha 字符串
            alpha_str = matches[2].replace('alpha', '')
            try:
                alpha_val = float(alpha_str)
            except ValueError:
                continue

            grouped[Dataset][alpha_val][Algorithm].append(np.array(res['server']['iid_accuracy']))

        # ===== 3. 开始绘图 =====
        for Dataset, alpha_dict in grouped.items():
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

            y_mins, y_maxs = [], []

            # 按照 alpha 从小到大排序 (0.01 -> 0.1 -> 1.0)
            for alpha_val in sorted(alpha_dict.keys()):
                algo_dict = alpha_dict[alpha_val]

                # 依次绘制 FedAvg 和 FedGeo
                for Algorithm in ['FedAvg', 'FedGeo']:
                    if Algorithm not in algo_dict:
                        continue

                    acc_list = algo_dict[Algorithm]
                    min_len = min(len(a) for a in acc_list)
                    acc_array = np.array([a[:min_len] for a in acc_list])

                    mean_acc = acc_array.mean(axis=0)
                    std_acc = acc_array.std(axis=0)

                    # 忽略前 20% 轮次用于 Y 轴极致收缩
                    ignore_idx = int(min_len * 0.20)
                    if len(mean_acc[ignore_idx:]) > 0:
                        y_mins.append(mean_acc[ignore_idx:].min())
                        y_maxs.append(mean_acc[ignore_idx:].max())

                    if Algorithm == 'FedGeo':
                        color = FEDGEO_GREEN_SHADES.get(alpha_val, FEDGEO_GREEN)
                    else:
                        color = baseline_alpha_colors.get(
                            alpha_val, sns.color_palette("bright")[7]
                        )
                    linestyle = algo_linestyles[Algorithm]

                    # 格式化图例文字
                    alpha_label = "1" if alpha_val == 1.0 else str(alpha_val)
                    label_str = f"{Algorithm} ($\\alpha$={alpha_label})"

                    # 加粗线条 (2.5)，确保在双栏排版缩放后依然清晰
                    linewidth = 2.5
                    zorder = 3 if Algorithm == 'FedGeo' else 2

                    ax.plot(mean_acc, label=label_str, linewidth=linewidth, color=color, linestyle=linestyle,
                            zorder=zorder)

                    # 阴影透明度设为 0.08，防止 6 条线交织时画面变脏
                    ax.fill_between(range(min_len), mean_acc - std_acc, mean_acc + std_acc,
                                    color=color, alpha=0.08, zorder=zorder - 0.5)

            # ===== 4. 图像细节设置 (TPAMI 级别规范) =====
            if y_mins and y_maxs:
                y_min_val = min(y_mins)
                y_max_val = max(y_maxs)
                # 留白压缩至 10%，让算法性能差异视觉化
                margin = (y_max_val - y_min_val) * 0.10
                ax.set_ylim(max(0.0, y_min_val - margin), y_max_val + margin)

            # 字号全面放大
            ax.set_xlabel('Communication Rounds', size=16)
            ax.set_ylabel('Testing Accuracy', size=16)
            ax.tick_params(axis='both', labelsize=14)

            # 双列图例防止遮挡，字号 14
            ax.legend(prop={'size': 14}, loc='lower right', ncol=2, framealpha=0.95)
            ax.grid(alpha=0.3)

            # 💡 核心修改：极致精简的纯数据集标题，字号调大到 22
            plt.suptitle(f"{Dataset}", size=22, fontweight='bold')

            # ===== 5. 自动保存 =====
            save_dir = os.path.join('..', 'plot', Dataset, 'multi_client')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"FedGeo_vs_FedAvg_{Dataset}_K50_C05.pdf")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ 成功保存极简标题的部分参与场景对比图: {save_path}")
