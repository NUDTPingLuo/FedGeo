# FedGeo

**Communication-Efficient Federated Learning via Implicit Geometric Steering**

PyTorch implementation of FedGeo and federated-learning baselines for image classification with heterogeneous client data. The repository includes training, bidirectional quantization, experiment sweeps, result visualization, and local-training profiling.

FedGeo reuses the previous round's decoded global model delta to synchronize client models and guide local optimization. It projects the normalized reference onto the subspace orthogonal to the local gradient, then adds this correction to the update direction. The steering operation does not require a client-specific control variate or a separate model-sized reference message. Setting `system.prox_beta: 0` disables steering; `system.q_bits: 32` selects full-precision communication.

This is a **single-machine FL simulation**: selected clients train sequentially within each communication round. It is not a multi-machine networking service.

## Repository structure

```text
.
├── fl_main.py                         # Single-experiment training entry point
├── run_multi_seed.py                  # Editable main-experiment sweep
├── run_supplementary_experiments.py    # CLI-based supplementary sweeps
├── profile_overhead.py                # CUDA local-training timing
├── config/
│   └── test_config.yaml               # Base experiment configuration
├── fed_baselines/                     # Client and server implementations
├── preprocessing/
│   └── baselines_dataloader.py        # Dataset loading and client partitions
├── utils/
│   ├── models.py                     # Model architectures
│   ├── fed_utils.py                  # Dataset/model dispatch
│   └── quantization.py               # Shared encoder, decoder, byte accounting
├── postprocessing/
│   ├── recorder.py                   # Result loading, metrics, plots
│   └── eval_main.py                  # Postprocessing CLI
├── data/                             # Local datasets; created/prepared separately
├── results/                          # Experiment records
├── cos_logs/                         # Reference/alignment diagnostics
└── plot/                             # Generated figures
```

Run the commands below **from the repository root** after activating the intended Python environment.

## Installation

Use a fresh Python 3.9 environment for the dependency versions below. An NVIDIA GPU is useful for training; `profile_overhead.py` requires CUDA because it uses CUDA timing events. The training classes select GPU 0 when available and otherwise fall back to CPU.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install PyTorch and torchvision using a matching build. For Linux/Windows with a compatible NVIDIA driver:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only Linux/Windows, use this command instead:

```bash
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
```

See the [official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/) for other platforms and builds. The versions above follow this repository's PyTorch version targets; they are not a claim that every platform has been tested.

Install the remaining packages imported by the training and plotting entry points:

```bash
python -m pip install numpy==1.21.6 Pillow==9.5.0 matplotlib==3.7.2 pandas==1.5.3 seaborn==0.12.2 PyYAML==6.0.1 tqdm==4.66.1
python -m pip check
python -c "import torch, torchvision, yaml, numpy, matplotlib, pandas, seaborn; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Dependency-file note:** the current `requirements.txt` contains incompatible duplicate matplotlib constraints and a literal `pip install ...` command. It is not directly usable with `pip install -r requirements.txt` as written. Use the explicit installation commands above. A fresh end-to-end installation and training run is still recommended on your target machine.

## Datasets and models

| Benchmark | Configuration value | Model | Classes | Preparation |
| --- | --- | --- | --- | --- |
| MNIST | `MNIST` | `LeNet` | 10 | Automatically downloaded into `data/` |
| Fashion-MNIST | `FashionMNIST` | `LeNet` | 10 | Automatically downloaded into `data/` |
| CIFAR-10 | `CIFAR10` | `CNN` | 10 | Automatically downloaded into `data/` |
| CIFAR-100 | `CIFAR100` | `CNN` | 100 | Automatically downloaded into `data/` |
| Tiny-ImageNet | `ImageNet` | `ResNet18` | 200 | Prepare locally as described below |

`ImageNet` is the code's identifier for **Tiny-ImageNet-200**, not full ImageNet. The ResNet constructor loads torchvision's default pretrained weights before replacing the input stem and classifier; its first use may download weights. Architecture definitions are in `utils/models.py`.

For Tiny-ImageNet, prepare an `ImageFolder`-compatible directory:

```text
data/tiny-imagenet-200/
├── train/
│   ├── <class-id>/images/*.JPEG
│   └── ...
└── val/
    ├── <class-id>/*.JPEG
    └── ...
```

Reorganize the original flat validation images into class folders using `val_annotations.txt`. Train and validation folders must use the same class IDs and class-to-index mapping. The loader does not perform this reorganization or download Tiny-ImageNet automatically.

Training uses a class-wise Dirichlet partition. Smaller `system.dirichlet_alpha` creates stronger label heterogeneity. The partitioner also transfers samples between clients to address small partitions according to `system.min_client_samples`; this is not an unconstrained Dirichlet draw. Choose feasible client counts and minimum sizes.

## Run one experiment

Edit `config/test_config.yaml`, then run:

```bash
python fl_main.py --config config/test_config.yaml
```

Example configuration:

```yaml
system:
  dataset: CIFAR10
  model: CNN
  num_client: 10
  num_round: 100
  dirichlet_alpha: 0.01
  min_client_samples: 256
  i_seed: 42
  prox_beta: 0.5
  q_bits: 32
  res_root: results
client:
  fed_algo: FedGeo
  lr: 0.01
  batch_size: 128
  num_local_epoch: 1
  momentum: 0
  fraction: 1.0
  fedcm_alpha: 0.1
  fedsol_rho: 0.1
  fedsol_temperature: 3.0
```

Important parameters:

| Key | Meaning |
| --- | --- |
| `client.fed_algo` | Algorithm identifier; see the list below |
| `client.lr` | Initial learning rate; training applies round-wise cosine annealing |
| `client.num_local_epoch` | Local epochs per selected client per round |
| `client.fraction` | Fraction selected for local training, in `(0, 1]` |
| `system.prox_beta` | FedGeo steering strength, despite the historical parameter name |
| `system.q_bits` | FedGeo/FedAvgQ communication precision; codec supports `2, 4, 8, 16, 32` |
| `system.i_seed` | Random seed for the run |
| `system.res_root` | Result root; use a separate directory for smoke tests |

`fl_main.py` accepts `FedAvg`, `FedGeo`, `FedAvgQ`, `FedCM`, `FedSOL`, `FedProx`, `FedNova`, and `Scaffold` (case-sensitive). The `FedSOL` implementation uses fixed perturbation strength and temperature-scaled KL divergence. FedDyn and FedSAM implementation files are also present and used by the profiler, but they are **not wired into this training entry point**.

For a quick smoke test, copy the configuration to a separate YAML file, choose `MNIST`/`LeNet`, reduce `num_round` to `2`, and set `res_root: results_smoke`. Pass that file to `fl_main.py`. Do not use the production result directory for short tests: round count is not included in result filenames.

## Run experiment sweeps

### Main comparisons

`run_multi_seed.py` exposes editable lists for datasets, algorithms, seeds, Dirichlet parameters, FedGeo bit widths, and steering strengths:

```bash
python run_multi_seed.py
```

The checked-in lists currently select all five benchmark datasets, seed `0`, FedGeo, three Dirichlet parameters, 32-bit communication, and `beta=0.5`: **15 jobs**. To reproduce five-seed comparisons, explicitly set:

```python
seeds = [0, 1, 42, 999, 2026]
algorithms = ["FedAvg", "FedGeo", "FedCM", "FedSOL", "FedProx", "FedNova", "Scaffold"]
```

With the other lists unchanged, this runs **525 jobs**. Learning rate, local epochs, rounds, and participation fraction are inherited from the base YAML.

**This script rewrites `config/test_config.yaml` in place** and stops on a failed subprocess. Back up that configuration before running; do not launch concurrent copies against the same file.

### Supplementary comparisons

The supplementary runner reads the base YAML but passes temporary configurations to training without modifying the original file. Its default suite contains:

| Comparison | Initial learning rate | Steering | Communication |
| --- | --- | --- | --- |
| FedGeo without steering | Base YAML rate | `beta=0` | 32-bit |
| FedAvg learning-rate sweep | `0.1`, `0.001` | Not applicable | 32-bit |
| FedGeo learning-rate sweep | `0.1`, `0.001` | `beta=0.5` | 32-bit |
| FedCM | Base YAML rate | Not applicable | 32-bit |
| FedSOL | Base YAML rate | Not applicable | 32-bit |
| FedAvgQ | Base YAML rate | Disabled | 2-, 4-, and 8-bit |

Keep the base rate at `0.01` for the corresponding controls. The `0.01` full-precision FedAvg/FedGeo references and within-FedGeo bit-width sweep must be run separately; they are not automatically added by this suite.

Preview the five-seed matrix before training:

```bash
python run_supplementary_experiments.py --datasets MNIST FashionMNIST CIFAR10 CIFAR100 --seeds 0 1 42 999 2026 --dry-run
```

Run the same matrix:

```bash
python run_supplementary_experiments.py --datasets MNIST FashionMNIST CIFAR10 CIFAR100 --seeds 0 1 42 999 2026 --skip-existing
```

These commands generate **600 jobs** before skipping: 4 datasets × 5 seeds × 3 Dirichlet parameters × 10 method settings. With no arguments, the currently checked-in runner selects seeds `1, 999` and the same four datasets, yielding **240 jobs**. Pass `--seeds 42 2026` for the two-seed subset instead.

The `ImageNet` entry is currently commented out in `DATASET_MODEL_MAP`, so `--datasets ImageNet` is rejected. To include Tiny-ImageNet, prepare its data and enable `"ImageNet": "ResNet18"` in that map. The corresponding five-dataset, five-seed default-method matrix contains **750 jobs**.

A separate high-learning-rate suite runs FedGeo at `lr=0.1`, `beta=0.1`, and 32 bits:

```bash
python run_supplementary_experiments.py --suite high-lr-beta-sweep --datasets MNIST FashionMNIST CIFAR10 CIFAR100 --seeds 0 1 42 999 2026 --high-lr-betas 0.1 --skip-existing
```

This is **60 jobs** before skipping. Its FedAvg and `beta=0.5` references come from the learning-rate sweep.

Other options include `--alphas`, `--lrs`, `--quant-bits`, `--rounds`, `--clients`, `--fedcm-alpha`, `--fedsol-rho`, and `--continue-on-error`. Inspect the full CLI with:

```bash
python run_supplementary_experiments.py --help
```

**Skipping is not checkpoint recovery.** `--skip-existing` checks file existence, not completion or configuration compatibility. Since results are written after every round, interrupted runs can leave partial files that will be skipped. Inspect and move incomplete records before rerunning.

## Results and visualization

Results are written to:

```text
results/<dataset>/alpha<alpha>/lr<lr>/beta<beta>/<algorithm>/<experiment-label>
```

The experiment label contains algorithm, dataset, alpha, learning rate, beta, and seed; compressed FedGeo/FedAvgQ runs add a bit-width tag. FedCM and FedSOL add their method-specific coefficient tags. Files contain JSON but have **no `.json` extension**. Directory names retain Python's string representation of configuration numbers, so `alpha1` and `alpha1.0` can both occur. The recorder understands the dataset-directory structure and deduplicates matching experiments; it does not search `results/test/`.

Each record includes the configuration and server traces such as `iid_accuracy`, `train_loss`, `uplink_bytes`, `downlink_bytes`, `communication_bytes`, `cumulative_communication_bytes`, and `round_time_seconds`. The historical `iid_accuracy` key records global test accuracy even when client training data are non-IID. `round_time_seconds` measures the whole simulated round, including training, aggregation, and evaluation, rather than one client's local training call.

Generate plots and tables from existing results:

```bash
python -m postprocessing.eval_main --plot main
python -m postprocessing.eval_main --plot table
python -m postprocessing.eval_main --plot beta0
python -m postprocessing.eval_main --plot lr
python -m postprocessing.eval_main --plot high-lr-beta
python -m postprocessing.eval_main --plot quant
python -m postprocessing.eval_main --plot beta
python -m postprocessing.eval_main --plot bit
```

Use `--plot all` to run all of these, or `--sys-res_root results_smoke` to select another result root. Figures are saved under `plot/`. Default comparison and ablation canvases are `(6, 5)` inches.

Postprocessing uses seeds **`0, 1, 42, 999, 2026`** by default. The table routine expects complete five-seed, 100-round series for each included configuration; short or incomplete results can raise an error. For a smaller seed set, call `Recorder` directly and pass the same seeds to loading and plotting:

```python
from postprocessing.recorder import Recorder

seeds = (42, 2026)
recorder = Recorder()
recorder.load_dataset_results("results", seeds=seeds)
recorder.plot(seeds=seeds)
```

FedGeo is emphasized with solid curves; comparison methods use dashed styles. The high-`lr` beta ablation highlights `beta=0.1` in green, while the standard beta ablation highlights `beta=0.5`.

Only load trusted experiment records. The legacy object decoder uses Python pickle for some serialized values.

## Local-training profiling

On a CUDA-capable machine:

```bash
python profile_overhead.py --config config/test_config.yaml --runs 30
```

The profiler performs warm-up calls and uses CUDA events to measure repeated complete local `train()` calls. It prints mean time, standard deviation, and overhead relative to FedAvg. Set `system.q_bits: 8` explicitly when profiling the 8-bit FedGeo configuration. These measurements are distinct from the whole-round timing traces produced by `fl_main.py` and do not measure real network latency or energy use.

## Reproducibility notes

- **Participation:** FedGeo and FedAvgQ broadcast each incremental payload to every client cache, even when only a subset trains. Missed downlinks and offline-client recovery are not simulated.
- **Quantization:** both directions use `utils/quantization.py`. Two- and four-bit codes are packed into bytes; eight-bit codes use signed integers. Byte accounting includes codec information, but does not represent network-protocol overhead or measured network traffic.
- **Output collisions:** round count, client count, local epochs, and several other settings are not encoded in filenames. Use distinct `res_root` directories when changing protocols or running smoke tests. Repeating the same output path overwrites the existing record.
- **Randomness:** the runner seeds Python, NumPy, and PyTorch. Exact numerical identity across devices and software versions is not guaranteed; keep configuration and environment records with reported results.
- **Scope:** available implementation files, training-entry-point support, and precomputed plotting baselines are not identical. Do not infer that every method shown in an existing figure can be selected in `fl_main.py`.

## Project reference

This code accompanies the manuscript **FedGeo: Communication-Efficient Federated Learning via Implicit Geometric Steering**. Publication metadata and a citation entry can be added when a public manuscript or publication identifier is available.
