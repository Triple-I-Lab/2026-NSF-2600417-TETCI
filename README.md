# FL-HE Framework

Implementation of the "Advancing Privacy and Accuracy with Federated Learning and Homomorphic Encryption" paper.

## Citation
 
```
Q. B. Phan, D. C. Nguyen, T. T. Doan, and T. T. Nguyen, "Advancing Privacy and Accuracy With Federated Learning and Homomorphic Encryption," in IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 10, no. 3, pp. 2416-2428, June 2026, doi: 10.1109/TETCI.2026.3670702.
```

## Requirements

```bash
pip install torch torchvision tenseal numpy matplotlib pyyaml scikit-learn
```

## Project Structure

```
root/
├── encryption.py       # CKKS scheme, homomorphic ops, noise bounds
├── gradient_method.py  # Diagonal Hessian approximation, Taylor gradient, convergence tracking
├── federated.py        # FL server, client, aggregation, Byzantine selection, FedAvg baseline
├── models.py           # CNN, Vision Transformer, EfficientNet-B0
├── datasets.py         # F-MNIST loader, Dirichlet partition, non-IID sampler
├── train.py            # Main entry point
└── config.yaml         # Default hyperparameters
```

Outputs are saved to `outputs/<run_tag>/` and inputs to `inputs/<run_tag>/`.

## Quick Start

```bash
# Fast test run
python train.py --mode simulate_he --clients 10 --rounds 20 --epochs 5

# Tenseal CKKS encryption
python train.py --mode tenseal --clients 10 --rounds 20

# With Byzantine attack
python train.py --attack fixed --attack_rate 0.2 --clients 10 --rounds 20

# Compare proposed vs FedAvg
python train.py --experiment compare --attack random --attack_rate 0.1

# Ablation study
python train.py --experiment ablation --clients 10 --rounds 20

# Full paper settings
python train.py --clients 100 --rounds 100 --epochs 100 --mode simulate_he
```

## CLI Flags

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--model` | `cnn`, `vit`, `efficientnet`, `all` | `cnn` | Model architecture |
| `--dataset` | `fmnist` | `fmnist` | Dataset |
| `--mode` | `simulate_he`, `tenseal` | `simulate_he` | Encryption mode |
| `--experiment` | `train`, `ablation`, `compare` | `train` | Experiment type |
| `--attack` | `none`, `fixed`, `random` | `none` | Byzantine attack type |
| `--attack_rate` | `0.0–0.5` | `0.1` | Fraction of Byzantine clients |
| `--clients` | int | `10` | Number of FL clients |
| `--rounds` | int | `20` | Federated rounds |
| `--epochs` | int | `5` | Local epochs per round |
| `--security` | `128`, `192`, `256` | `128` | CKKS security level (bits) |
| `--run_tag` | str | auto | Custom run identifier |
| `--resume` | str | — | Resume from existing run tag |
| `--seed` | int | `42` | Random seed |

## Self-Tests

Each file includes 3 built-in test cases runnable independently:

```bash
python encryption.py     [--mode simulate_he | tenseal]
python gradient_method.py [--mode simulate_he | tenseal]
python federated.py      [--mode simulate_he | tenseal]
python models.py         [--mode simulate_he | tenseal]
python datasets.py       [--mode simulate_he | tenseal]
python train.py          [--mode simulate_he | tenseal]
```

## Encryption Modes

**`simulate_he`** — injects statistically equivalent Gaussian noise without TenSEAL. Fast, suitable for debugging and quick experiments. Results track tenseal CKKS closely.

**`tenseal`** — CKKS encryption via TenSEAL. Slower but cryptographically correct. Recommended for results.

## Output Structure

```
inputs/  <run_tag>/  config.json
outputs/ <run_tag>/  metrics.json
                     checkpoints/  <model>_server.pt
                                   <model>_final.pt
                     plots/
```

## Notes

- F-MNIST downloads automatically on first run to `inputs/raw_data/`
- X-ray dataset requires manual download from Kaggle 
- CIFAR-10 downloads automatically via torchvision
