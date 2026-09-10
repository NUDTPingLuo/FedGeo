"""One-click runner for the five reviewer-requested comparison groups.

Default scope:
  1. FedGeo beta=0 with the original learning rate;
  2. FedAvg and FedGeo beta=0.5 with learning rates 0.1 and 0.001;
  3. FedCM with the original learning rate;
  4. FedSOL-fixed with the original learning rate;
  5. FedAvg-Q at 2/4/8 bits with the original learning rate.

An additional ``high-lr-beta-sweep`` suite runs FedGeo at lr=0.1 with
smaller beta values.  It is kept separate from the default reviewer suite so
the diagnostic experiments can be launched without repeating other methods.

The original config file is only read.  Each subprocess receives a temporary
YAML file, so interrupted runs do not leave ``config/test_config.yaml`` in a
surprising state.  Existing result paths and directory structure are retained.
"""

import argparse
import copy
import importlib.util
import itertools
import os
import subprocess
import sys
import tempfile
import time

import yaml


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "test_config.yaml")
DATASET_MODEL_MAP = {
    "MNIST": "LeNet",
    "FashionMNIST": "LeNet",
    "CIFAR10": "CNN",
    "CIFAR100": "CNN",
    # "ImageNet": "ResNet18",
}


def format_number(value):
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all supplementary comparison experiments")
    parser.add_argument(
        "--suite", choices=["supplementary", "high-lr-beta-sweep"],
        default="supplementary",
        help="Experiment group to run (default: supplementary)",
    )
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASET_MODEL_MAP),
                        default=list(DATASET_MODEL_MAP))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1,999])
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.01, 0.1, 1])
    parser.add_argument("--lrs", nargs="+", type=float, default=[0.1, 0.001])
    parser.add_argument(
        "--high-lr-betas", nargs="+", type=float, default=[0.1],
        help=(
            "FedGeo beta values for --suite high-lr-beta-sweep. "
            "lr is fixed at 0.1; beta=0.5 is omitted by default because it "
            "already belongs to the supplementary LR sweep."
        ),
    )
    parser.add_argument("--quant-bits", nargs="+", type=int, default=[2, 4, 8],
                        choices=[2, 4, 8, 16, 32],
                        help="FedAvg-Q widths; defaults to 2, 4 and 8 bits")
    parser.add_argument("--rounds", type=int, help="Override the base number of rounds")
    parser.add_argument("--clients", type=int, help="Override the base number of clients")
    parser.add_argument("--fedcm-alpha", type=float, default=0.1)
    parser.add_argument("--fedsol-rho", type=float, default=0.1)
    parser.add_argument("--dry-run", action="store_true", help="Print the experiment matrix only")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Resume a run by skipping result files that already exist")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def experiment_key(experiment):
    return (
        experiment["algo"], experiment["lr"], experiment["beta"],
        experiment["q_bits"], experiment.get("fedcm_alpha"),
        experiment.get("fedsol_rho")
    )


def build_method_matrix(base_lr, learning_rates, quant_bits, fedcm_alpha, fedsol_rho):
    # Methods that should follow the original experimental learning rate.
    experiments = [
        {"algo": "FedGeo", "lr": base_lr, "beta": 0.0, "q_bits": 32},
        {"algo": "FedCM", "lr": base_lr, "beta": 0.0, "q_bits": 32,
         "fedcm_alpha": fedcm_alpha},
        {"algo": "FedSOL", "lr": base_lr, "beta": 0.0, "q_bits": 32,
         "fedsol_rho": fedsol_rho},
    ]

    # Only FedAvg and the normal FedGeo (beta=0.5) receive the LR sweep.
    for lr in learning_rates:
        experiments.extend([
            {"algo": "FedGeo", "lr": lr, "beta": 0.5, "q_bits": 32},
            {"algo": "FedAvg", "lr": lr, "beta": 0.0, "q_bits": 32},
        ])

    # Bit ablation is performed only for FedAvg-Q, using the original LR.
    for bits in quant_bits:
        experiments.append({
            "algo": "FedAvgQ", "lr": base_lr, "beta": 0.0, "q_bits": bits
        })

    # ``quant_bits=32`` overlaps with the full-precision FedGeo reference.
    unique = {}
    for experiment in experiments:
        unique[experiment_key(experiment)] = experiment
    return list(unique.values())


def build_high_lr_beta_matrix(beta_values):
    """Build the dedicated FedGeo beta sweep at the fixed high LR of 0.1."""
    experiments = []
    for beta in beta_values:
        if beta < 0:
            raise ValueError("FedGeo beta must be non-negative, got {}".format(beta))
        experiments.append({
            "algo": "FedGeo", "lr": 0.1, "beta": beta, "q_bits": 32
        })

    unique = {}
    for experiment in experiments:
        unique[experiment_key(experiment)] = experiment
    return list(unique.values())


def result_filename(config):
    system = config["system"]
    client = config["client"]
    algo = client["fed_algo"]
    tags = [
        algo, system["dataset"], "alpha{}".format(format_number(system["dirichlet_alpha"])),
        "lr{}".format(format_number(client["lr"])),
        "beta{}".format(format_number(system["prox_beta"])),
        "seed{}".format(system["i_seed"]),
    ]
    bits = int(system.get("q_bits", 32))
    if algo in ("FedGeo", "FedAvgQ") and bits != 32:
        tags.append("bit{}".format(bits))
    if algo == "FedCM":
        tags.append("cmalpha{}".format(format_number(client.get("fedcm_alpha", 0.1))))
    if algo == "FedSOL":
        tags.append("rho{}".format(format_number(client.get("fedsol_rho", 0.1))))
    return "[{}]".format(",".join("'{}'".format(tag) for tag in tags))


def result_candidate_paths(config):
    """Return current and normalized locations for resume checks only."""
    system = config["system"]
    client = config["client"]
    result_root = system.get("res_root", "results")
    if not os.path.isabs(result_root):
        result_root = os.path.join(PROJECT_ROOT, result_root)

    # fl_main.py currently uses str(value) for directories, while previously
    # merged results may use normalized names such as alpha1 instead of
    # alpha1.0.  Check both without changing where new results are saved.
    alpha_labels = {
        str(system["dirichlet_alpha"]), format_number(system["dirichlet_alpha"])
    }
    lr_labels = {str(client["lr"]), format_number(client["lr"])}
    beta_labels = {str(system["prox_beta"]), format_number(system["prox_beta"])}
    filename = result_filename(config)

    paths = []
    for alpha, lr, beta in itertools.product(alpha_labels, lr_labels, beta_labels):
        paths.append(os.path.join(
            result_root,
            system["dataset"],
            "alpha" + alpha,
            "lr" + lr,
            "beta" + beta,
            client["fed_algo"],
            filename,
        ))
    return paths


def prepare_config(base_config, dataset, seed, alpha, experiment, args):
    config = copy.deepcopy(base_config)
    config.setdefault("system", {})
    config.setdefault("client", {})
    config["system"].update({
        "dataset": dataset,
        "model": DATASET_MODEL_MAP[dataset],
        "i_seed": seed,
        "dirichlet_alpha": alpha,
        "prox_beta": experiment["beta"],
        "q_bits": experiment["q_bits"],
    })
    config["client"].update({
        "fed_algo": experiment["algo"],
        "lr": experiment["lr"],
        "fedcm_alpha": experiment.get("fedcm_alpha", args.fedcm_alpha),
        "fedsol_rho": experiment.get("fedsol_rho", args.fedsol_rho),
        "fedsol_temperature": 3.0,
    })
    if args.rounds is not None:
        config["system"]["num_round"] = args.rounds
    if args.clients is not None:
        config["system"]["num_client"] = args.clients
    return config


def main():
    args = parse_args()
    if not args.dry_run and importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "PyTorch is not available in this Python environment. Activate the "
            "project's PyTorch environment, then run this script again."
        )
    with open(BASE_CONFIG_PATH, "r") as stream:
        base_config = yaml.safe_load(stream)

    if args.suite == "high-lr-beta-sweep":
        methods = build_high_lr_beta_matrix(args.high_lr_betas)
    else:
        methods = build_method_matrix(
            base_config["client"]["lr"], args.lrs, args.quant_bits,
            args.fedcm_alpha, args.fedsol_rho
        )
    jobs = []
    for dataset, seed, alpha, experiment in itertools.product(
            args.datasets, args.seeds, args.alphas, methods):
        jobs.append(prepare_config(base_config, dataset, seed, alpha, experiment, args))

    print("Experiment suite: {}".format(args.suite))
    print("Supplementary experiment jobs: {}".format(len(jobs)))
    failed = []
    started = time.time()

    with tempfile.TemporaryDirectory(prefix="fedgeo_supp_") as temp_dir:
        runtime_config = os.path.join(temp_dir, "experiment.yaml")
        for index, config in enumerate(jobs, start=1):
            system = config["system"]
            client = config["client"]
            label = (
                "[{}/{}] {} / {} / seed={} / alpha={} / lr={} / bit={} / beta={}"
            ).format(
                index, len(jobs), system["dataset"], client["fed_algo"],
                system["i_seed"], system["dirichlet_alpha"], client["lr"],
                system["q_bits"], system["prox_beta"]
            )

            result_paths = result_candidate_paths(config)
            if args.skip_existing and any(os.path.exists(path) for path in result_paths):
                print(label + " -- skipped (exists)")
                continue
            print(label)
            if args.dry_run:
                continue

            with open(runtime_config, "w") as stream:
                yaml.safe_dump(config, stream, default_flow_style=False, sort_keys=False)
            try:
                subprocess.run(
                    [sys.executable, "fl_main.py", "--config", runtime_config],
                    cwd=PROJECT_ROOT, check=True
                )
            except subprocess.CalledProcessError as error:
                failed.append((label, error.returncode))
                if not args.continue_on_error:
                    raise

    elapsed = time.time() - started
    print("Finished in {:.1f} minutes; failed jobs: {}".format(elapsed / 60.0, len(failed)))
    for label, return_code in failed:
        print("FAILED (exit {}): {}".format(return_code, label))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
