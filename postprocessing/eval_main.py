#!/usr/bin/env python3
import argparse
try:
    from .recorder import Recorder, DEFAULT_RESULTS_ROOT, DEFAULT_PLOT_SEEDS
except ImportError:
    from recorder import Recorder, DEFAULT_RESULTS_ROOT, DEFAULT_PLOT_SEEDS


def fed_args():
    """
    Arguments for running postprocessing on FedD3
    :return: Arguments for postprocessing on FedD3
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-rr', '--sys-res_root', type=str, default=DEFAULT_RESULTS_ROOT,
        help='Results root containing the five dataset folders (default: project/results)'
    )
    parser.add_argument(
        '--plot', choices=[
            'main', 'table', 'beta', 'beta0', 'lr', 'high-lr-beta',
            'bit', 'quant', 'all'
        ],
        default='all', help='Which comparison to generate'
    )

    args = parser.parse_args()
    return args


def res_eval():
    """
    Main function for result evaluation
    """
    args = fed_args()

    recorder = Recorder()

    recorder.load_dataset_results(
        args.sys_res_root, seeds=DEFAULT_PLOT_SEEDS
    )

    if args.plot in ('main', 'all'):
        recorder.plot(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('table', 'all'):
        recorder.print_table_metrics(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('beta', 'all'):
        recorder.plot_beta_ablation(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('beta0', 'all'):
        recorder.plot_fedgeo_beta0_vs_fedavg(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('lr', 'all'):
        recorder.plot_lr_ablation_fedavg_fedgeo(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('high-lr-beta', 'all'):
        recorder.plot_high_lr_beta_ablation(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('bit', 'all'):
        recorder.plot_bit_ablation(seeds=DEFAULT_PLOT_SEEDS)
    if args.plot in ('quant', 'all'):
        recorder.plot_quantization_ablation_fedavgq_fedgeo(seeds=DEFAULT_PLOT_SEEDS)


if __name__ == "__main__":
    res_eval()
