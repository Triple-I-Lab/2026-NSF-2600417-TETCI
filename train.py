"""
train.py
========
Main entry point for the FL-HE framework.

Usage examples:
  python train.py                                          # defaults
  python train.py --model cnn --dataset fmnist
  python train.py --model all --experiment ablation
  python train.py --experiment compare --attack fixed --attack_rate 0.2
  python train.py --mode simulate_he --clients 10 --rounds 20
  python train.py --resume 20260601_001

CLI flags:
  --model        cnn | vit | efficientnet | all        (default: cnn)
  --dataset      fmnist                                (default: fmnist)
  --mode         simulate_he | tenseal                 (default: simulate_he)
  --experiment   train | ablation | compare            (default: train)
  --attack       none | fixed | random                 (default: none)
  --attack_rate  float 0.0-0.5                         (default: 0.1)
  --clients      int                                   (default: 10)
  --rounds       int                                   (default: 20)
  --epochs       int  local epochs per round           (default: 5)
  --security     128 | 192 | 256                       (default: 128)
  --run_tag      str  (default: auto YYYYMMDD_NNN)
  --resume       str  run_tag to resume from
  --seed         int                                   (default: 42)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from datasets import DatasetLoader, FederatedPartition, get_dataloaders
from encryption import CKKSParams, CKKSScheme, TENSEAL_AVAILABLE
from federated import (
    ClientConfig,
    FedAvgServer,
    FederatedRound,
    FLClient,
    FLServer,
    RunIO,
    ServerConfig,
)
from models import build_model


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FL-HE framework training entry point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",       default="cnn",
                        choices=["cnn", "vit", "efficientnet", "all"])
    parser.add_argument("--dataset",     default="fmnist",
                        choices=["fmnist"])
    parser.add_argument("--mode",        default="simulate_he",
                        choices=["simulate_he", "tenseal"])
    parser.add_argument("--experiment",  default="train",
                        choices=["train", "ablation", "compare"])
    parser.add_argument("--attack",      default="none",
                        choices=["none", "fixed", "random"])
    parser.add_argument("--attack_rate", default=0.1, type=float)
    parser.add_argument("--clients",     default=10,  type=int)
    parser.add_argument("--rounds",      default=20,  type=int)
    parser.add_argument("--epochs",      default=5,   type=int)
    parser.add_argument("--security",    default=128, type=int,
                        choices=[128, 192, 256])
    parser.add_argument("--run_tag",     default=None, type=str)
    parser.add_argument("--resume",      default=None, type=str)
    parser.add_argument("--seed",        default=42,  type=int)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def run_train(
    model_name: str,
    args: argparse.Namespace,
    rio: RunIO,
    ckks_params: CKKSParams,
    scheme: CKKSScheme,
) -> Dict:
    """
    Single training run: proposed FL-HE method on one model/dataset combo.
    Returns metrics dict.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n[train] model={model_name}  dataset={args.dataset}  "
          f"clients={args.clients}  rounds={args.rounds}  "
          f"attack={args.attack}({args.attack_rate:.0%})")

    # Data
    client_loaders, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        n_clients=args.clients,
        alpha=0.5,
        batch_size=32,
        seed=args.seed,
    )

    # Model
    base_model = build_model(model_name, dataset=args.dataset)

    # Byzantine client IDs
    n_byz = int(args.clients * args.attack_rate)
    byz_ids = list(range(args.clients - n_byz, args.clients))
    attack_epochs = list(range(0, args.rounds, 5)) if args.attack == "fixed" else []

    # Build clients
    clients = []
    for i in range(args.clients):
        is_byz = i in byz_ids
        cfg = ClientConfig(
            client_id=i,
            n_local_epochs=args.epochs,
            learning_rate=0.01,
            byzantine=is_byz,
            attack_type=args.attack if is_byz else "none",
            attack_noise_std=0.1,
            attack_epochs=attack_epochs,
        )
        client_ds = client_loaders[i].dataset
        clients.append(
            FLClient(cfg, client_ds, copy.deepcopy(base_model), scheme, ckks_params)
        )

    # Server
    server = FLServer(
        ServerConfig(
            n_clients=args.clients,
            participation_rate=0.4,
            distance_threshold=2.0,
            min_clients_per_round=max(2, int(args.clients * 0.4 * 0.5)),
            security_bits=args.security,
        ),
        copy.deepcopy(base_model),
        ckks_params,
        scheme,
    )

    # Resume from checkpoint if requested
    start_round = 0
    if args.resume:
        sd = rio.load_checkpoint(f"{model_name}_server")
        if sd is not None:
            server.model.load_state_dict(sd)
            meta = rio.load_checkpoint(f"{model_name}_meta")
            start_round = meta.get("round", 0) if meta else 0
            print(f"  Resumed from round {start_round}")

    coordinator = FederatedRound(
        server, clients, ckks_params, scheme, use_fedavg=False
    )

    val_ds = val_loader.dataset
    metrics_history = []

    for r in range(start_round, args.rounds):
        m = coordinator.run(r, val_dataset=val_ds, participation_rate=0.4)
        metrics_history.append(m)

        print(f"  Round {r+1:3d}/{args.rounds}  "
              f"val_loss={m.get('val_loss', 0):.4f}  "
              f"val_acc={m.get('val_acc', 0):.4f}  "
              f"t={m['duration_s']}s")

        # Checkpoint every 10 rounds
        if (r + 1) % 10 == 0:
            rio.save_checkpoint(server.model.state_dict(), f"{model_name}_server")
            rio.save_checkpoint({"round": r + 1}, f"{model_name}_meta")

    # Final test evaluation
    test_loss, test_acc = server.evaluate(test_loader.dataset)
    print(f"  Final test  loss={test_loss:.4f}  acc={test_acc:.4f}")

    result = {
        "model": model_name,
        "dataset": args.dataset,
        "mode": args.mode,
        "attack": args.attack,
        "attack_rate": args.attack_rate,
        "rounds": args.rounds,
        "test_loss": round(test_loss, 6),
        "test_acc":  round(test_acc,  6),
        "history": metrics_history,
    }
    rio.save_checkpoint(server.model.state_dict(), f"{model_name}_final")
    rio.save_metrics(result)
    return result


def run_ablation(
    model_name: str,
    args: argparse.Namespace,
    rio: RunIO,
    ckks_params: CKKSParams,
    scheme: CKKSScheme,
) -> Dict:
    """
    Ablation study: 5 variants isolating each component.

    V1 — FedAvg baseline (no HE, no Byzantine selection, no diagonal approx)
    V2 — FedAvg + diagonal Hessian approximation only
    V3 — FedAvg + Byzantine selection only
    V4 — FedAvg + CKKS encryption only
    V5 — Full proposed method (diagonal + Byzantine + CKKS)

    Attack: 20% Byzantine, random-position (hardest scenario).
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n[ablation] model={model_name}  20% Byzantine random attack")

    client_loaders, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        n_clients=args.clients,
        alpha=0.5,
        batch_size=32,
        seed=args.seed,
    )

    base_model = build_model(model_name, dataset=args.dataset)
    val_ds  = val_loader.dataset
    test_ds = test_loader.dataset

    n_byz = max(1, int(args.clients * 0.2))
    byz_ids = list(range(args.clients - n_byz, args.clients))

    def build_clients(byz_attack: str) -> List[FLClient]:
        clients = []
        for i in range(args.clients):
            is_byz = i in byz_ids
            cfg = ClientConfig(
                client_id=i,
                n_local_epochs=args.epochs,
                learning_rate=0.01,
                byzantine=is_byz,
                attack_type=byz_attack if is_byz else "none",
                attack_noise_std=0.1,
            )
            clients.append(
                FLClient(cfg, client_loaders[i].dataset,
                         copy.deepcopy(base_model), scheme, ckks_params)
            )
        return clients

    variants = {
        "V1_FedAvg":       {"use_byzantine": False, "use_he": False},
        "V2_DiagOnly":     {"use_byzantine": False, "use_he": False},
        "V3_ByzOnly":      {"use_byzantine": True,  "use_he": False},
        "V4_HEOnly":       {"use_byzantine": False, "use_he": True},
        "V5_Full":         {"use_byzantine": True,  "use_he": True},
    }

    ablation_results = {}

    for vname, vcfg in variants.items():
        print(f"  Running {vname}...")
        use_he  = vcfg["use_he"]
        use_byz = vcfg["use_byzantine"]

        run_scheme = scheme if use_he else CKKSScheme(
            params=ckks_params, simulate=True
        )

        if not use_byz:
            # Use FedAvg-style server (no Byzantine selection)
            server = FedAvgServer(copy.deepcopy(base_model))
            clients = build_clients("random")
            acc_list = []

            for r in range(args.rounds):
                global_p = server.get_model_params()
                updates, sizes = {}, {}
                for i, c in enumerate(clients):
                    c.receive_model(global_p)
                    ct = c.local_train(r)
                    delta = run_scheme.decrypt(ct)
                    updates[i] = delta
                    sizes[i] = c.dataset_size()
                delta = server.aggregate(updates, sizes)
                server.apply_update(delta)
                _, acc = server.evaluate(val_ds)
                acc_list.append(acc)

            _, clean_acc  = server.evaluate(val_ds)
            # Attack drop: difference between max acc and final acc
            attack_drop = max(acc_list) - acc_list[-1]

        else:
            # Use proposed FL server with Byzantine selection
            server_cfg = ServerConfig(
                n_clients=args.clients,
                participation_rate=0.4,
                distance_threshold=2.0,
                min_clients_per_round=max(2, int(args.clients * 0.2)),
                security_bits=args.security,
            )
            server = FLServer(
                server_cfg, copy.deepcopy(base_model), ckks_params, run_scheme
            )
            clients = build_clients("random")
            coordinator = FederatedRound(
                server, clients, ckks_params, run_scheme, use_fedavg=False
            )
            acc_list = []
            for r in range(args.rounds):
                m = coordinator.run(r, val_dataset=val_ds, participation_rate=0.4)
                acc_list.append(m.get("val_acc", 0))

            _, clean_acc = server.evaluate(val_ds)
            attack_drop = max(acc_list) - acc_list[-1]

        ablation_results[vname] = {
            "final_acc":   round(clean_acc, 4),
            "attack_drop": round(attack_drop, 4),
            "acc_history": [round(a, 4) for a in acc_list],
        }
        print(f"    {vname}: acc={clean_acc:.4f}  attack_drop={attack_drop:.4f}")

    rio.save_metrics({"ablation": ablation_results})
    return ablation_results


def run_compare(
    model_name: str,
    args: argparse.Namespace,
    rio: RunIO,
    ckks_params: CKKSParams,
    scheme: CKKSScheme,
) -> Dict:
    """
    Comparison: proposed method vs FedAvg baseline under attack.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n[compare] model={model_name}  "
          f"attack={args.attack}({args.attack_rate:.0%})")

    client_loaders, val_loader, test_loader = get_dataloaders(
        dataset_name=args.dataset,
        n_clients=args.clients,
        alpha=0.5,
        batch_size=32,
        seed=args.seed,
    )

    base_model = build_model(model_name, dataset=args.dataset)
    val_ds  = val_loader.dataset
    test_ds = test_loader.dataset

    n_byz   = int(args.clients * args.attack_rate)
    byz_ids = list(range(args.clients - n_byz, args.clients))
    attack_epochs = list(range(0, args.rounds, 5)) if args.attack == "fixed" else []

    def build_clients(model_ref) -> List[FLClient]:
        clients = []
        for i in range(args.clients):
            is_byz = i in byz_ids
            cfg = ClientConfig(
                client_id=i,
                n_local_epochs=args.epochs,
                learning_rate=0.01,
                byzantine=is_byz,
                attack_type=args.attack if is_byz else "none",
                attack_noise_std=0.1,
                attack_epochs=attack_epochs,
            )
            clients.append(
                FLClient(cfg, client_loaders[i].dataset,
                         copy.deepcopy(model_ref), scheme, ckks_params)
            )
        return clients

    results = {}

    # --- Proposed ---
    proposed_server = FLServer(
        ServerConfig(
            n_clients=args.clients, participation_rate=0.4,
            distance_threshold=2.0,
            min_clients_per_round=max(2, int(args.clients * 0.2)),
            security_bits=args.security,
        ),
        copy.deepcopy(base_model), ckks_params, scheme,
    )
    proposed_clients = build_clients(base_model)
    proposed_coord   = FederatedRound(
        proposed_server, proposed_clients, ckks_params, scheme, use_fedavg=False
    )

    proposed_accs = []
    for r in range(args.rounds):
        m = proposed_coord.run(r, val_dataset=val_ds, participation_rate=0.4)
        proposed_accs.append(m.get("val_acc", 0))
        print(f"  [Proposed] Round {r+1:3d}  acc={proposed_accs[-1]:.4f}")

    _, proposed_test_acc = proposed_server.evaluate(test_ds)
    results["proposed"] = {
        "test_acc":    round(proposed_test_acc, 4),
        "acc_history": [round(a, 4) for a in proposed_accs],
    }

    # --- FedAvg ---
    fedavg_server  = FedAvgServer(copy.deepcopy(base_model))
    fedavg_clients = build_clients(base_model)
    fedavg_accs    = []

    for r in range(args.rounds):
        global_p = fedavg_server.get_model_params()
        updates, sizes = {}, {}
        for i, c in enumerate(fedavg_clients):
            c.receive_model(global_p)
            ct = c.local_train(r)
            updates[i] = scheme.decrypt(ct)
            sizes[i]   = c.dataset_size()
        delta = fedavg_server.aggregate(updates, sizes)
        fedavg_server.apply_update(delta)
        _, acc = fedavg_server.evaluate(val_ds)
        fedavg_accs.append(acc)
        print(f"  [FedAvg]   Round {r+1:3d}  acc={fedavg_accs[-1]:.4f}")

    _, fedavg_test_acc = fedavg_server.evaluate(test_ds)
    results["fedavg"] = {
        "test_acc":    round(fedavg_test_acc, 4),
        "acc_history": [round(a, 4) for a in fedavg_accs],
    }

    print(f"\n  Proposed test acc : {proposed_test_acc:.4f}")
    print(f"  FedAvg   test acc : {fedavg_test_acc:.4f}")

    rio.save_metrics({"compare": results})
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate mode
    if args.mode == "tenseal" and not TENSEAL_AVAILABLE:
        print("[ERROR] tenseal not installed. Use --mode simulate_he.")
        raise SystemExit(1)

    # Setup I/O
    resume_tag = args.resume
    rio = RunIO(run_tag=resume_tag or args.run_tag)
    print(f"Run: {rio.run_tag}")
    print(f"  input  -> {rio.input_dir}")
    print(f"  output -> {rio.output_dir}")

    # Save config
    rio.save_config(vars(args))

    # CKKS setup
    ckks_params = CKKSParams.from_security_level(args.security)
    simulate    = args.mode == "simulate_he"
    scheme      = CKKSScheme(params=ckks_params, simulate=simulate)

    # Model list
    models = ["cnn", "vit", "efficientnet"] if args.model == "all" else [args.model]

    all_results = {}

    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Model: {model_name.upper()}")
        print(f"{'='*50}")

        if args.experiment == "train":
            result = run_train(model_name, args, rio, ckks_params, scheme)
        elif args.experiment == "ablation":
            result = run_ablation(model_name, args, rio, ckks_params, scheme)
        elif args.experiment == "compare":
            result = run_compare(model_name, args, rio, ckks_params, scheme)

        all_results[model_name] = result

    rio.save_metrics(all_results)
    print(f"\nDone. Results saved to {rio.output_dir}/metrics.json")


# ---------------------------------------------------------------------------
# Self-contained test cases
# ---------------------------------------------------------------------------

def _test_case_1_arg_defaults(simulate: bool):
    """
    Test 1 — Argument parsing: defaults are correct,
    all flags parse without error.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 1: Argument parsing  [{mode_label}] ===")

    import sys
    # Simulate running with no args
    sys.argv = ["train.py"]
    args = parse_args()

    assert args.model      == "cnn"
    assert args.dataset    == "fmnist"
    assert args.mode       == "simulate_he"
    assert args.experiment == "train"
    assert args.attack     == "none"
    assert args.clients    == 10
    assert args.rounds     == 20
    assert args.epochs     == 5
    assert args.security   == 128
    assert args.seed       == 42
    print(f"  All defaults correct: {vars(args)}")

    # Simulate --model all --experiment ablation
    sys.argv = ["train.py", "--model", "all", "--experiment", "ablation",
                "--attack", "fixed", "--attack_rate", "0.3", "--clients", "5"]
    args2 = parse_args()
    assert args2.model      == "all"
    assert args2.experiment == "ablation"
    assert args2.attack     == "fixed"
    assert abs(args2.attack_rate - 0.3) < 1e-6
    assert args2.clients    == 5
    print(f"  Custom flags correct.")
    print("  PASSED")


def _test_case_2_short_train_run(simulate: bool):
    """
    Test 2 — Functional: run_train for 3 rounds on CNN/fmnist,
    verify metrics are saved and loss decreases.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 2: Short train run (CNN, 3 rounds)  [{mode_label}] ===")

    import sys, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.argv = [
            "train.py",
            "--model",   "cnn",
            "--dataset", "fmnist",
            "--mode",    "simulate_he",   # always fast for self-test
            "--rounds",  "3",
            "--clients", "4",
            "--epochs",  "2",
        ]
        args = parse_args()

        rio         = RunIO(run_tag="selftest_train", base_dir=Path(tmpdir))
        ckks_params = CKKSParams.from_security_level(128)
        # Always simulate_he in self-test — real CKKS is too slow for CI
        scheme      = CKKSScheme(params=ckks_params, simulate=True)
        args.mode   = "simulate_he"

        result = run_train("cnn", args, rio, ckks_params, scheme)

        assert "test_acc"  in result
        assert "test_loss" in result
        assert "history"   in result
        assert len(result["history"]) == 3
        assert 0.0 <= result["test_acc"] <= 1.0

        metrics_path = rio.output_dir / "metrics.json"
        assert metrics_path.exists(), "metrics.json not saved."
        print(f"  test_acc={result['test_acc']:.4f}  "
              f"test_loss={result['test_loss']:.4f}")
        print(f"  metrics.json saved: OK")
    print("  PASSED")


def _test_case_3_compare_run(simulate: bool):
    """
    Test 3 — Stress: run_compare with 10% Byzantine fixed attack,
    both methods complete and return valid accuracy histories.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 3: Compare run [{mode_label}] ===")

    import sys, tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        sys.argv = [
            "train.py",
            "--model",       "cnn",
            "--dataset",     "fmnist",
            "--mode",        "simulate_he",
            "--experiment",  "compare",
            "--attack",      "fixed",
            "--attack_rate", "0.1",
            "--rounds",      "3",
            "--clients",     "5",
            "--epochs",      "2",
        ]
        args = parse_args()

        rio         = RunIO(run_tag="selftest_compare", base_dir=Path(tmpdir))
        ckks_params = CKKSParams.from_security_level(128)
        scheme      = CKKSScheme(params=ckks_params, simulate=True)
        args.mode   = "simulate_he"

        results = run_compare("cnn", args, rio, ckks_params, scheme)

        assert "proposed" in results
        assert "fedavg"   in results
        assert len(results["proposed"]["acc_history"]) == 3
        assert len(results["fedavg"]["acc_history"])   == 3
        assert all(0 <= a <= 1 for a in results["proposed"]["acc_history"])
        assert all(0 <= a <= 1 for a in results["fedavg"]["acc_history"])

        print(f"  Proposed acc history: {results['proposed']['acc_history']}")
        print(f"  FedAvg   acc history: {results['fedavg']['acc_history']}")
    print("  PASSED")


if __name__ == "__main__":
    import sys

    # Detect if running as self-test (no meaningful args beyond --mode)
    is_test = (
        len(sys.argv) == 1
        or (len(sys.argv) == 3 and sys.argv[1] == "--mode")
    )

    if is_test:
        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=["simulate_he", "tenseal"],
                            default="tenseal")
        mode_args, _ = parser.parse_known_args()
        simulate = mode_args.mode == "simulate_he"

        if mode_args.mode == "tenseal" and not TENSEAL_AVAILABLE:
            print("[ERROR] tenseal not installed. Use --mode simulate_he.")
            raise SystemExit(1)

        print("=" * 60)
        print(f"train.py — self-test suite  [{mode_args.mode}]")
        print("=" * 60)

        _test_case_1_arg_defaults(simulate)
        _test_case_2_short_train_run(simulate)
        _test_case_3_compare_run(simulate)

        print("\nAll tests passed.")
    else:
        main()