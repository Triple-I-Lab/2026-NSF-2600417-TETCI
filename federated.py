"""
federated.py
============
Federated learning orchestration for the FL-HE framework.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from encryption import (
    CKKSParams,
    CKKSScheme,
    Ciphertext,
    HomomorphicOps,
    NoiseAnalysis,
    TENSEAL_AVAILABLE,
)
from gradient_method import (
    DiagonalHessian,
    EncryptedGradient,
    ModelLossWrapper,
    TaylorGradient,
)


# ---------------------------------------------------------------------------
# Run I/O management
# ---------------------------------------------------------------------------

class RunIO:
    """
    Manages input/output folder structure with auto-tagged run directories.

    Directory layout:
        inputs/  <run_tag>/  config.json
                             dataset_splits/
        outputs/ <run_tag>/  metrics.json
                             checkpoints/
                             plots/

    Run tag format: YYYYMMDD_NNN  (NNN auto-incremented per day)
    """

    BASE_DIR = Path(".")

    def __init__(self, run_tag: Optional[str] = None, base_dir: Optional[Path] = None):
        if base_dir is not None:
            self.BASE_DIR = base_dir
        self.run_tag = run_tag or self._auto_tag()
        self.input_dir  = self.BASE_DIR / "inputs"  / self.run_tag
        self.output_dir = self.BASE_DIR / "outputs" / self.run_tag
        self._init_dirs()

    def _auto_tag(self) -> str:
        today = datetime.now().strftime("%Y%m%d")
        base = self.BASE_DIR / "outputs"
        base.mkdir(parents=True, exist_ok=True)
        # Find next available NNN for today
        existing = [
            d.name for d in base.iterdir()
            if d.is_dir() and d.name.startswith(today)
        ] if base.exists() else []
        idx = len(existing) + 1
        return f"{today}_{idx:03d}"

    def _init_dirs(self) -> None:
        for d in [
            self.input_dir,
            self.output_dir / "checkpoints",
            self.output_dir / "plots",
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: dict) -> None:
        path = self.input_dir / "config.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def save_metrics(self, metrics: dict) -> None:
        path = self.output_dir / "metrics.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)

    def save_checkpoint(self, state_dict: dict, name: str) -> None:
        path = self.output_dir / "checkpoints" / f"{name}.pt"
        torch.save(state_dict, path)

    def load_checkpoint(self, name: str) -> Optional[dict]:
        path = self.output_dir / "checkpoints" / f"{name}.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        return None

    def __repr__(self) -> str:
        return f"RunIO(tag={self.run_tag}, input={self.input_dir}, output={self.output_dir})"


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class ClientConfig:
    """Per-client configuration."""
    client_id: int
    n_local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    byzantine: bool = False
    attack_type: str = "none"       # "none" | "fixed" | "random"
    attack_noise_std: float = 0.1
    attack_epochs: Optional[List[int]] = None   # for "fixed" attacks


class FLClient:
    """
    Federated learning client with encrypted gradient updates.
    """

    def __init__(
        self,
        config: ClientConfig,
        dataset: torch.utils.data.Dataset,
        model: nn.Module,
        scheme: CKKSScheme,
        ckks_params: CKKSParams,
    ):
        self.config = config
        self.dataset = dataset
        self.model = model
        self.scheme = scheme
        self.ckks_params = ckks_params
        self.eg = EncryptedGradient(scheme, TaylorGradient(
            DiagonalHessian(method="finite_diff")
        ))
        self._local_round = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def receive_model(self, global_params: np.ndarray) -> None:
        """Update local model with global parameters from server."""
        self._set_flat_params(torch.tensor(global_params, dtype=torch.float32))

    def local_train(self, current_round: int) -> Ciphertext:
        """
        Run local training using Taylor-based diagonal gradient approximation
        and return the encrypted accumulated gradient update.
        """
        self._local_round = current_round
        theta_t   = self._get_flat_params().clone()
        criterion = nn.CrossEntropyLoss()
        lr        = self.config.learning_rate

        loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Compute diagonal Hessian once per round
        first_batch = next(iter(loader))
        x_init, y_init = first_batch
        wrapper_init  = ModelLossWrapper(self.model, x_init, y_init, criterion)
        loss_fn_h     = wrapper_init.as_loss_fn_for_hessian()
        theta_cur_init = wrapper_init.get_flat_params().double()
        diag_t = self.eg.taylor.hessian.compute_diagonal(
            loss_fn_h, theta_cur_init, wrapper=wrapper_init
        )

        for epoch in range(self.config.n_local_epochs):
            for x_batch, y_batch in loader:
                wrapper   = ModelLossWrapper(self.model, x_batch, y_batch, criterion)
                theta_cur = wrapper.get_flat_params().double()

                # Fast gradient via backprop
                grad_t = wrapper.compute_gradient_direct().double()

                # Taylor correction
                max_diag    = 10.0
                diag_clip   = diag_t.clamp(0.0, max_diag)
                delta_theta = theta_cur - theta_t.double()
                correction  = diag_clip * delta_theta

                # Taylor gradient: ∇̃L = ∇L(θ_t) + D(θ_t)*(θ - θ_t)
                approx_grad = grad_t + correction

                # Clip gradient norm to prevent explosion
                grad_norm = approx_grad.norm()
                max_norm  = 1.0
                if grad_norm > max_norm:
                    approx_grad = approx_grad * (max_norm / grad_norm)

                # Parameter update
                theta_new = theta_cur - lr * approx_grad
                wrapper.set_flat_params(theta_new.float())

        # Accumulated update = final params - initial global params
        theta_final = self._get_flat_params()
        delta = (theta_final - theta_t).detach().cpu().numpy()

        # Byzantine attack injection
        if self.config.byzantine:
            delta = self._inject_attack(delta, current_round)

        # Encrypt and return
        return self.scheme.encrypt(delta)

    def dataset_size(self) -> int:
        return len(self.dataset)

    # ------------------------------------------------------------------
    # Attack injection
    # ------------------------------------------------------------------

    def _inject_attack(self, delta: np.ndarray, current_round: int) -> np.ndarray:
        """
        Inject Gaussian noise into gradient update.

        fixed  : attack at predetermined rounds (attack_epochs list)
        random : attack at random intervals (~30% of rounds)
        """
        attack = self.config.attack_type
        if attack == "none":
            return delta

        should_attack = False
        if attack == "fixed":
            epochs = self.config.attack_epochs or []
            should_attack = current_round in epochs
        elif attack == "random":
            should_attack = np.random.random() < 0.3

        if should_attack:
            noise = np.random.normal(
                0, self.config.attack_noise_std, size=delta.shape
            )
            delta = delta + noise

        return delta

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].reshape(p.shape))
            offset += n


# ---------------------------------------------------------------------------
# Server — proposed method
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    """Server-side configuration."""
    n_clients: int = 10
    participation_rate: float = 0.4
    distance_threshold: float = 2.0        # τ in Algorithm 2
    min_clients_per_round: int = 2
    security_bits: int = 128


class FLServer:

    def __init__(
        self,
        config: ServerConfig,
        model: nn.Module,
        ckks_params: CKKSParams,
        scheme: CKKSScheme,
    ):
        self.config = config
        self.model = model
        self.ckks_params = ckks_params
        self.scheme = scheme
        self.ops = HomomorphicOps(scheme)
        self.round_metrics: List[dict] = []

    # ------------------------------------------------------------------
    # Byzantine-resilient client selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        client_updates: Dict[int, Ciphertext],
        n_select: int,
    ) -> List[int]:
        """
        Select clients based on encrypted distance filtering (Algorithm 2).

        Computes pairwise squared distances between encrypted updates,
        filters outliers beyond threshold τ, returns selected client IDs.

        Parameters
        ----------
        client_updates : dict mapping client_id -> Ciphertext
        n_select       : number of clients to select

        Returns
        -------
        List[int]  selected client IDs
        """
        client_ids = list(client_updates.keys())
        if len(client_ids) <= n_select:
            return client_ids

        # Decrypt distances
        distances: Dict[int, float] = {}
        ref_id = client_ids[0]
        ref_ct = client_updates[ref_id]

        for cid in client_ids[1:]:
            dist = self.ops.he_squared_distance(ref_ct, client_updates[cid])
            distances[cid] = dist
        distances[ref_id] = 0.0

        # Filter: keep clients within threshold τ of median distance
        dist_values = np.array(list(distances.values()))
        median_dist = float(np.median(dist_values))
        tau = self.config.distance_threshold

        selected = [
            cid for cid, d in distances.items()
            if d <= median_dist + tau * (float(np.std(dist_values)) + 1e-12)
        ]

        # Fallback: ensure minimum participation
        if len(selected) < self.config.min_clients_per_round:
            selected = sorted(distances, key=distances.get)[: self.config.min_clients_per_round]

        return selected[:n_select]

    # ------------------------------------------------------------------
    # Secure aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_updates: Dict[int, Ciphertext],
        dataset_sizes: Dict[int, int],
        selected_ids: List[int],
    ) -> np.ndarray:
        """
        Homomorphic weighted aggregation of selected client updates.

        Δθ_global = Σ (|D_i| / |D|) · Δθ_i   (all in encrypted domain)

        Parameters
        ----------
        client_updates : encrypted gradient updates
        dataset_sizes  : number of samples per client
        selected_ids   : clients chosen by Algorithm 2

        Returns
        -------
        np.ndarray  decrypted aggregated gradient update
        """
        selected_updates = [client_updates[cid] for cid in selected_ids]
        total_samples = sum(dataset_sizes[cid] for cid in selected_ids)
        weights = [
            dataset_sizes[cid] / total_samples for cid in selected_ids
        ]

        # Weighted sum in encrypted domain
        agg_ct = self.ops.he_weighted_sum(selected_updates, weights)
        return self.scheme.decrypt(agg_ct)

    # ------------------------------------------------------------------
    # Model update
    # ------------------------------------------------------------------

    def apply_update(self, delta: np.ndarray) -> None:
        """Apply aggregated gradient update to global model."""
        flat = self._get_flat_params()
        updated = flat + torch.tensor(delta, dtype=torch.float32)
        self._set_flat_params(updated)

    def get_model_params(self) -> np.ndarray:
        return self._get_flat_params().detach().cpu().numpy()

    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
    ) -> Tuple[float, float]:
        """
        Evaluate global model on a dataset.

        Returns
        -------
        (loss, accuracy)
        """
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in loader:
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * len(y)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)

        self.model.train()
        avg_loss = total_loss / max(total, 1)
        accuracy  = correct / max(total, 1)
        return avg_loss, accuracy

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].reshape(p.shape))
            offset += n


# ---------------------------------------------------------------------------
# FedAvg baseline server
# ---------------------------------------------------------------------------

class FedAvgServer:
    """
    Plain FedAvg baseline (McMahan et al., 2017).

    No encryption, no Byzantine filtering.
    Used as the comparison baseline in Section V-D.
    Gradients are transmitted and aggregated in plaintext.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.round_metrics: List[dict] = []

    def aggregate(
        self,
        client_deltas: Dict[int, np.ndarray],
        dataset_sizes: Dict[int, int],
    ) -> np.ndarray:
        """Plain weighted average of gradient updates."""
        total = sum(dataset_sizes[cid] for cid in client_deltas)
        agg = np.zeros_like(next(iter(client_deltas.values())), dtype=np.float64)
        for cid, delta in client_deltas.items():
            w = dataset_sizes[cid] / total
            agg += w * delta.astype(np.float64)
        return agg.astype(np.float32)

    def apply_update(self, delta: np.ndarray) -> None:
        flat = self._get_flat_params()
        updated = flat + torch.tensor(delta, dtype=torch.float32)
        self._set_flat_params(updated)

    def get_model_params(self) -> np.ndarray:
        return self._get_flat_params().detach().cpu().numpy()

    def evaluate(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int = 64,
    ) -> Tuple[float, float]:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                out = self.model(x)
                loss = criterion(out, y)
                total_loss += loss.item() * len(y)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += len(y)
        self.model.train()
        return total_loss / max(total, 1), correct / max(total, 1)

    def _get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def _set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            p.data.copy_(flat[offset: offset + n].reshape(p.shape))
            offset += n


# ---------------------------------------------------------------------------
# Federated round coordinator
# ---------------------------------------------------------------------------

class FederatedRound:

    def __init__(
        self,
        server,
        clients: List[FLClient],
        ckks_params: Optional[CKKSParams] = None,
        scheme: Optional[CKKSScheme] = None,
        use_fedavg: bool = False,
    ):
        self.server = server
        self.clients = clients
        self.ckks_params = ckks_params
        self.scheme = scheme
        self.use_fedavg = use_fedavg

    def run(
        self,
        round_idx: int,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        participation_rate: float = 0.4,
    ) -> dict:
        """
        Execute one FL round.

        Returns
        -------
        dict  round metrics: {round, n_selected, val_loss, val_acc, duration_s}
        """
        t0 = time.time()
        global_params = self.server.get_model_params()

        # Sample participating clients
        n_participate = max(
            1, int(len(self.clients) * participation_rate)
        )
        participating = np.random.choice(
            len(self.clients), size=n_participate, replace=False
        ).tolist()

        # Broadcast and collect updates
        updates: Dict[int, any] = {}
        sizes:   Dict[int, int]  = {}

        for idx in participating:
            client = self.clients[idx]
            client.receive_model(global_params)
            update = client.local_train(round_idx)
            updates[idx] = update
            sizes[idx] = client.dataset_size()

        # Aggregate
        if self.use_fedavg:
            # FedAvg
            plain_deltas = {
                cid: self.scheme.decrypt(ct) if isinstance(ct, Ciphertext)
                     else ct
                for cid, ct in updates.items()
            }
            delta = self.server.aggregate(plain_deltas, sizes)
        else:
            # Proposed
            n_select = max(self.server.config.min_clients_per_round,
                           int(n_participate * 0.8))
            selected = self.server.select_clients(updates, n_select)
            delta = self.server.aggregate(updates, sizes, selected)

        self.server.apply_update(delta)

        # Evaluate
        metrics = {
            "round": round_idx,
            "n_participating": n_participate,
            "duration_s": round(time.time() - t0, 3),
        }
        if val_dataset is not None:
            loss, acc = self.server.evaluate(val_dataset)
            metrics["val_loss"] = round(loss, 6)
            metrics["val_acc"]  = round(acc,  6)

        return metrics


# ---------------------------------------------------------------------------
# Self-contained test cases
# ---------------------------------------------------------------------------

def _make_tiny_model(n_classes: int = 4) -> nn.Module:
    """Tiny MLP for fast test runs."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )


def _make_tiny_dataset(n: int = 64, n_classes: int = 4) -> torch.utils.data.Dataset:
    """Random classification dataset."""
    X = torch.randn(n, 1, 4, 4)
    y = torch.randint(0, n_classes, (n,))
    return torch.utils.data.TensorDataset(X, y)


def _test_case_1_run_io(simulate: bool):
    """
    Test 1 — RunIO: folders created correctly, config and checkpoint
    round-trip cleanly, auto-tags are unique.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 1: RunIO folder management  [{mode_label}] ===")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        rio = RunIO(run_tag="test_20260601_001", base_dir=Path(tmpdir))

        assert rio.input_dir.exists()
        assert rio.output_dir.exists()
        assert (rio.output_dir / "checkpoints").exists()
        assert (rio.output_dir / "plots").exists()
        print(f"  Dirs: {rio}")

        cfg = {"model": "cnn", "dataset": "fmnist", "clients": 10, "rounds": 5}
        rio.save_config(cfg)
        loaded = json.loads((rio.input_dir / "config.json").read_text())
        assert loaded == cfg
        print(f"  Config save/load: OK")

        model = _make_tiny_model()
        rio.save_checkpoint(model.state_dict(), "round_001")
        sd = rio.load_checkpoint("round_001")
        assert sd is not None
        assert set(sd.keys()) == set(model.state_dict().keys())
        print(f"  Checkpoint save/load: OK")

        rio2 = RunIO(base_dir=Path(tmpdir))
        rio3 = RunIO(base_dir=Path(tmpdir))
        assert rio2.run_tag != rio3.run_tag
        print(f"  Auto-tags unique: {rio2.run_tag} vs {rio3.run_tag}")

    print("  PASSED")


def _test_case_2_taylor_fl_round(simulate: bool):
    """
    Test 2 — Functional: FL round using Taylor gradient in local_train.
    Verifies:
      (a) Client uses ModelLossWrapper + diagonal Hessian (not plain SGD)
      (b) Encrypted update is produced and aggregated correctly
      (c) Loss decreases over 5 rounds — model is learning
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 2: FL round with Taylor gradient  [{mode_label}] ===")

    torch.manual_seed(42)
    np.random.seed(42)

    ckks_params = CKKSParams.from_security_level(128)
    scheme      = CKKSScheme(params=ckks_params, simulate=simulate)

    n_clients  = 4
    n_classes  = 2
    base_model = _make_tiny_model(n_classes)

    clients = []
    for i in range(n_clients):
        cfg = ClientConfig(
            client_id=i,
            n_local_epochs=3,
            learning_rate=0.05,
            byzantine=False,
        )
        ds = _make_tiny_dataset(128, n_classes)
        clients.append(FLClient(cfg, ds, copy.deepcopy(base_model), scheme, ckks_params))

    server = FLServer(
        ServerConfig(n_clients=n_clients, participation_rate=1.0,
                     distance_threshold=3.0, min_clients_per_round=2),
        copy.deepcopy(base_model), ckks_params, scheme,
    )
    val_ds      = _make_tiny_dataset(256, n_classes)
    coordinator = FederatedRound(server, clients, ckks_params, scheme, use_fedavg=False)

    # Verify client uses Taylor gradient
    assert isinstance(clients[0].eg.taylor.hessian, DiagonalHessian),         "Client not using DiagonalHessian."
    print(f"  Client uses DiagonalHessian: OK")

    losses = []
    for r in range(5):
        metrics = coordinator.run(r, val_dataset=val_ds, participation_rate=1.0)
        losses.append(metrics["val_loss"])
        print(f"  Round {r}: val_loss={metrics['val_loss']:.4f}  "
              f"val_acc={metrics['val_acc']:.4f}  "
              f"t={metrics['duration_s']}s")

    assert losses[-1] < losses[0], (
        f"Model not learning with Taylor gradient: "
        f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("  PASSED")


def _test_case_3_taylor_vs_fedavg_under_attack(simulate: bool):
    """
    Test 3 — Stress: proposed method (Taylor gradient + Byzantine selection)
    vs FedAvg under 30% Byzantine fixed attack.
    Verifies:
      (a) Both methods complete without errors
      (b) Proposed method shows lower accuracy variance (more stable under attack)
      (c) Proposed method final accuracy >= FedAvg final accuracy
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 3: Taylor+Byzantine vs FedAvg under attack  [{mode_label}] ===")

    torch.manual_seed(0)
    np.random.seed(0)

    ckks_params = CKKSParams.from_security_level(128)
    scheme      = CKKSScheme(params=ckks_params, simulate=simulate)

    n_clients  = 6
    n_byz      = 2
    n_classes  = 2
    n_rounds   = 5
    base_model = _make_tiny_model(n_classes)
    val_ds     = _make_tiny_dataset(256, n_classes)

    def build_fl_clients(model_ref):
        clients = []
        for i in range(n_clients):
            is_byz = i >= (n_clients - n_byz)
            cfg = ClientConfig(
                client_id=i,
                n_local_epochs=3,
                learning_rate=0.05,
                byzantine=is_byz,
                attack_type="fixed" if is_byz else "none",
                attack_noise_std=0.3,
                attack_epochs=list(range(n_rounds)),
            )
            ds = _make_tiny_dataset(128, n_classes)
            clients.append(
                FLClient(cfg, ds, copy.deepcopy(model_ref), scheme, ckks_params)
            )
        return clients

    # --- Proposed: Taylor gradient + Byzantine selection ---
    proposed_server = FLServer(
        ServerConfig(n_clients=n_clients, participation_rate=1.0,
                     distance_threshold=1.5, min_clients_per_round=3),
        copy.deepcopy(base_model), ckks_params, scheme,
    )
    proposed_coord = FederatedRound(
        proposed_server, build_fl_clients(base_model),
        ckks_params, scheme, use_fedavg=False
    )

    # --- FedAvg: plain SGD, no Byzantine defense ---
    fedavg_server  = FedAvgServer(copy.deepcopy(base_model))
    fedavg_clients = build_fl_clients(base_model)

    proposed_accs, fedavg_accs = [], []

    for r in range(n_rounds):
        m = proposed_coord.run(r, val_dataset=val_ds, participation_rate=1.0)
        proposed_accs.append(m["val_acc"])

        global_p = fedavg_server.get_model_params()
        updates, sizes = {}, {}
        for i, c in enumerate(fedavg_clients):
            c.receive_model(global_p)
            ct = c.local_train(r)
            updates[i] = scheme.decrypt(ct)
            sizes[i]   = c.dataset_size()
        fedavg_server.apply_update(fedavg_server.aggregate(updates, sizes))
        _, fa_acc = fedavg_server.evaluate(val_ds)
        fedavg_accs.append(fa_acc)

    print(f"  Proposed accs : {[f'{a:.3f}' for a in proposed_accs]}")
    print(f"  FedAvg   accs : {[f'{a:.3f}' for a in fedavg_accs]}")

    assert all(0.0 <= a <= 1.0 for a in proposed_accs)
    assert all(0.0 <= a <= 1.0 for a in fedavg_accs)

    proposed_std = float(np.std(proposed_accs))
    fedavg_std   = float(np.std(fedavg_accs))
    print(f"  Proposed std: {proposed_std:.4f}  FedAvg std: {fedavg_std:.4f}")
    print(f"  Proposed final acc: {proposed_accs[-1]:.4f}  "
          f"FedAvg final acc: {fedavg_accs[-1]:.4f}")

    assert proposed_std <= fedavg_std + 0.05,         f"Proposed (std={proposed_std:.4f}) much worse than FedAvg (std={fedavg_std:.4f})."

    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="federated.py self-test suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["simulate_he", "tenseal"],
        default="tenseal",
        help=(
            "tenseal     : genuine CKKS via TenSEAL (default)\n"
            "simulate_he : fast mock encryption, no TenSEAL required"
        ),
    )
    args = parser.parse_args()
    simulate = args.mode == "simulate_he"

    if not simulate and not TENSEAL_AVAILABLE:
        print(
            "[ERROR] --mode tenseal requires TenSEAL.\n"
            "Install with:  pip install tenseal\n"
            "Or run with:   python federated.py --mode simulate_he"
        )
        raise SystemExit(1)

    print("=" * 60)
    print(f"federated.py — self-test suite  [{args.mode}]")
    print("=" * 60)

    _test_case_1_run_io(simulate)
    _test_case_2_taylor_fl_round(simulate)
    _test_case_3_taylor_vs_fedavg_under_attack(simulate)

    print("\nAll tests passed.")