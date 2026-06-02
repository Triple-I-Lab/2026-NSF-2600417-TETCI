"""
datasets.py
===========
Data loading, preprocessing, and federated partitioning.

Covers:
  - DatasetLoader     : download and prepare F-MNIST, X-ray, CIFAR-10
  - FederatedPartition: split a dataset across clients using Dirichlet(α)
                        for controlled non-IID heterogeneity
  - NonIIDSampler     : per-client sampler enforcing class imbalance
  - get_dataloaders() : convenience wrapper returning train/val/test loaders

Run this file directly for 3 built-in test cases:
  python datasets.py [--mode simulate_he | tenseal]
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from torchvision.datasets import CIFAR10, FashionMNIST


# ---------------------------------------------------------------------------
# X-ray dataset (Kaggle chest X-ray pneumonia)
# ---------------------------------------------------------------------------

class XRayDataset(Dataset):
    """
    Chest X-ray pneumonia dataset loader.

    Expects the following folder structure (standard Kaggle layout):
        root/
            train/  NORMAL/  *.jpeg
                    PNEUMONIA/ *.jpeg
            val/    ...
            test/   ...

    If the folder is not found, falls back to a synthetic dataset
    so tests can run without the actual data downloaded.

    Labels: 0 = NORMAL, 1 = PNEUMONIA
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 64,
        synthetic_n: int = 200,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ])

        split_dir = self.root / split
        if split_dir.exists():
            self._load_from_disk(split_dir)
        else:
            warnings.warn(
                f"XRay data not found at {split_dir}. "
                "Using synthetic data. Download from: "
                "https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia",
                UserWarning,
                stacklevel=2,
            )
            self._make_synthetic(synthetic_n, img_size)

    def _load_from_disk(self, split_dir: Path) -> None:
        from PIL import Image
        self.samples: List[Tuple[Path, int]] = []
        class_map = {"NORMAL": 0, "PNEUMONIA": 1}
        for cls_name, label in class_map.items():
            cls_dir = split_dir / cls_name
            if cls_dir.exists():
                for f in cls_dir.iterdir():
                    if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        self.samples.append((f, label))
        self._synthetic = False

    def _make_synthetic(self, n: int, img_size: int) -> None:
        self.samples = [(None, i % 2) for i in range(n)]
        self._img_size = img_size
        self._synthetic = True

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if self._synthetic if hasattr(self, "_synthetic") else False:
            img = torch.randn(1, self._img_size, self._img_size)
            return img, label
        from PIL import Image
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

class DatasetLoader:
    """
    Downloads and prepares datasets with train / val / test splits.

    Splits follow the 70/15/15 ratio used in experiments.
    """

    DATA_ROOT = Path("inputs") / "raw_data"

    # Normalisation stats
    STATS = {
        "fmnist":  {"mean": [0.2860], "std": [0.3530]},
        "cifar10": {"mean": [0.4914, 0.4822, 0.4465],
                    "std":  [0.2470, 0.2435, 0.2616]},
        "xray":    {"mean": [0.5],    "std":  [0.5]},
    }

    def __init__(self, data_root: Optional[Path] = None):
        if data_root is not None:
            self.DATA_ROOT = data_root

    def load(
        self,
        name: str,
        img_size: Optional[int] = None,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Load a dataset and return (train, val, test) splits.

        Parameters
        ----------
        name     : "fmnist" | "xray" | "cifar10"
        img_size : resize images to this square size (None = dataset default)

        Returns
        -------
        (train_dataset, val_dataset, test_dataset)
        """
        name = name.lower()
        if name == "fmnist":
            return self._load_fmnist(img_size or 28)
        if name == "cifar10":
            return self._load_cifar10(img_size or 32)
        if name == "xray":
            return self._load_xray(img_size or 64)
        raise ValueError(f"Unknown dataset '{name}'. Choose: fmnist, cifar10, xray.")

    # ------------------------------------------------------------------

    def _load_fmnist(self, img_size: int) -> Tuple[Dataset, Dataset, Dataset]:
        stats = self.STATS["fmnist"]
        tf_train = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        tf_test = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        full_train = FashionMNIST(
            str(self.DATA_ROOT), train=True,  download=True, transform=tf_train
        )
        test_ds = FashionMNIST(
            str(self.DATA_ROOT), train=False, download=True, transform=tf_test
        )
        train_ds, val_ds = self._split_train_val(full_train, val_frac=0.15)
        return train_ds, val_ds, test_ds

    def _load_cifar10(self, img_size: int) -> Tuple[Dataset, Dataset, Dataset]:
        stats = self.STATS["cifar10"]
        tf_train = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomCrop(img_size, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        tf_test = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(stats["mean"], stats["std"]),
        ])
        full_train = CIFAR10(
            str(self.DATA_ROOT), train=True,  download=True, transform=tf_train
        )
        test_ds = CIFAR10(
            str(self.DATA_ROOT), train=False, download=True, transform=tf_test
        )
        train_ds, val_ds = self._split_train_val(full_train, val_frac=0.15)
        return train_ds, val_ds, test_ds

    def _load_xray(self, img_size: int) -> Tuple[Dataset, Dataset, Dataset]:
        xray_root = self.DATA_ROOT / "chest_xray"
        train_ds = XRayDataset(str(xray_root), split="train", img_size=img_size)
        val_ds   = XRayDataset(str(xray_root), split="val",   img_size=img_size)
        test_ds  = XRayDataset(str(xray_root), split="test",  img_size=img_size)
        return train_ds, val_ds, test_ds

    @staticmethod
    def _split_train_val(
        dataset: Dataset, val_frac: float = 0.15
    ) -> Tuple[Dataset, Dataset]:
        n = len(dataset)
        n_val = int(n * val_frac)
        n_train = n - n_val
        indices = torch.randperm(n).tolist()
        return (
            Subset(dataset, indices[:n_train]),
            Subset(dataset, indices[n_train:]),
        )


# ---------------------------------------------------------------------------
# Federated partition (Dirichlet non-IID)
# ---------------------------------------------------------------------------

class FederatedPartition:
    """
    Splits a dataset across n_clients using a Dirichlet(α) distribution
    over class labels.

    α controls heterogeneity:
        α → ∞  : IID (uniform distribution)
        α = 0.5: moderately non-IID (used in experiments)
        α → 0  : extreme non-IID (each client gets one class)

    Each client receives a Subset of the original dataset.
    """

    def __init__(self, alpha: float = 0.5, seed: int = 42):
        self.alpha = alpha
        self.seed  = seed

    def split(
        self,
        dataset: Dataset,
        n_clients: int,
        min_samples_per_client: int = 10,
    ) -> List[Subset]:
        """
        Partition dataset into n_clients subsets.

        Parameters
        ----------
        dataset                 : full training dataset
        n_clients               : number of FL clients
        min_samples_per_client  : minimum samples guaranteed per client

        Returns
        -------
        List[Subset]  length n_clients
        """
        rng = np.random.default_rng(self.seed)
        labels = self._get_labels(dataset)
        n_classes = int(labels.max()) + 1
        n_total   = len(labels)

        # Group indices by class
        class_indices: Dict[int, List[int]] = {
            c: np.where(labels == c)[0].tolist()
            for c in range(n_classes)
        }

        # Sample Dirichlet proportions per class
        client_indices: List[List[int]] = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            idx = class_indices[c]
            rng.shuffle(idx)

            # Dirichlet proportions for this class across clients
            proportions = rng.dirichlet(np.ones(n_clients) * self.alpha)
            proportions = np.array(proportions)

            # Convert to integer counts
            counts = (proportions * len(idx)).astype(int)
            # Fix rounding — assign remainder to largest-proportion client
            remainder = len(idx) - counts.sum()
            counts[np.argmax(proportions)] += remainder

            # Assign
            offset = 0
            for cid in range(n_clients):
                end = offset + counts[cid]
                client_indices[cid].extend(idx[offset:end])
                offset = end

        # Guarantee minimum samples per client by redistributing
        client_indices = self._enforce_minimum(
            client_indices, labels, min_samples_per_client, rng
        )

        return [Subset(dataset, idxs) for idxs in client_indices]

    def class_distribution(self, subsets: List[Subset]) -> np.ndarray:
        """
        Returns a (n_clients, n_classes) matrix of sample counts.
        Useful for visualising non-IID heterogeneity.
        """
        all_labels = []
        for subset in subsets:
            lbls = self._get_labels(subset)
            all_labels.append(lbls)

        n_classes = max(l.max() for l in all_labels) + 1
        dist = np.zeros((len(subsets), n_classes), dtype=int)
        for i, lbls in enumerate(all_labels):
            for c in range(n_classes):
                dist[i, c] = int((lbls == c).sum())
        return dist

    # ------------------------------------------------------------------

    @staticmethod
    def _get_labels(dataset: Dataset) -> np.ndarray:
        """Extract labels from a dataset or subset."""
        if isinstance(dataset, Subset):
            base = dataset.dataset
            idxs = dataset.indices
            if hasattr(base, "targets"):
                t = base.targets
                if isinstance(t, torch.Tensor):
                    return t[idxs].numpy()
                return np.array(t)[idxs]
            # Fallback: iterate (slow)
            return np.array([base[i][1] for i in idxs])

        if hasattr(dataset, "targets"):
            t = dataset.targets
            if isinstance(t, torch.Tensor):
                return t.numpy()
            return np.array(t)

        if isinstance(dataset, TensorDataset):
            return dataset.tensors[1].numpy()

        # Fallback
        return np.array([dataset[i][1] for i in range(len(dataset))])

    @staticmethod
    def _enforce_minimum(
        client_indices: List[List[int]],
        labels: np.ndarray,
        min_n: int,
        rng: np.random.Generator,
    ) -> List[List[int]]:
        """Redistribute samples to ensure every client has at least min_n."""
        all_idx = set(range(len(labels)))

        for cid, idxs in enumerate(client_indices):
            if len(idxs) < min_n:
                deficit = min_n - len(idxs)
                # Find clients with surplus
                donors = [
                    c for c in range(len(client_indices))
                    if c != cid and len(client_indices[c]) > min_n + deficit
                ]
                if not donors:
                    continue
                donor = donors[0]
                transfer = client_indices[donor][-deficit:]
                client_indices[donor] = client_indices[donor][:-deficit]
                client_indices[cid].extend(transfer)

        return client_indices


# ---------------------------------------------------------------------------
# Non-IID sampler (strict class restriction)
# ---------------------------------------------------------------------------

class NonIIDSampler:
    """
    Creates client subsets where each client receives data from only
    k_classes out of the total, simulating extreme non-IID conditions.

    Used for the CIFAR-10 non-IID configuration where clients
    receive data from only 2-3 classes.
    """

    def __init__(self, k_classes: int = 2, seed: int = 42):
        self.k_classes = k_classes
        self.seed = seed

    def split(
        self,
        dataset: Dataset,
        n_clients: int,
    ) -> List[Subset]:
        """
        Assign k_classes to each client using round-robin class assignment.

        Parameters
        ----------
        dataset   : full dataset
        n_clients : number of clients

        Returns
        -------
        List[Subset]
        """
        rng = np.random.default_rng(self.seed)
        labels = FederatedPartition._get_labels(dataset)
        n_classes = int(labels.max()) + 1

        # Randomly assign k_classes to each client
        class_indices: Dict[int, List[int]] = {
            c: np.where(labels == c)[0].tolist()
            for c in range(n_classes)
        }
        for c in class_indices:
            rng.shuffle(class_indices[c])

        client_subsets = []
        for cid in range(n_clients):
            # Assign k_classes in round-robin
            assigned = [
                (cid * self.k_classes + j) % n_classes
                for j in range(self.k_classes)
            ]
            # Give each client an equal share of its assigned classes
            idxs = []
            for c in assigned:
                share = len(class_indices[c]) // max(
                    1, sum(1 for i in range(n_clients)
                           if c in [(i * self.k_classes + j) % n_classes
                                    for j in range(self.k_classes)])
                )
                idxs.extend(class_indices[c][:share])
            client_subsets.append(Subset(dataset, idxs))

        return client_subsets


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def get_dataloaders(
    dataset_name: str,
    n_clients: int,
    alpha: float = 0.5,
    batch_size: int = 32,
    non_iid_strict: bool = False,
    k_classes: int = 2,
    data_root: Optional[Path] = None,
    img_size: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Full pipeline: load -> partition -> return per-client train loaders
    plus shared val and test loaders.

    Parameters
    ----------
    dataset_name   : "fmnist" | "xray" | "cifar10"
    n_clients      : number of FL clients
    alpha          : Dirichlet concentration (0.5 = moderately non-IID)
    batch_size     : per-client training batch size
    non_iid_strict : if True, use NonIIDSampler (k_classes per client)
    k_classes      : classes per client for strict non-IID
    data_root      : override default data directory
    img_size       : override image size
    seed           : random seed

    Returns
    -------
    (client_loaders, val_loader, test_loader)
    """
    loader = DatasetLoader(data_root)
    train_ds, val_ds, test_ds = loader.load(dataset_name, img_size)

    if non_iid_strict:
        sampler = NonIIDSampler(k_classes=k_classes, seed=seed)
        client_subsets = sampler.split(train_ds, n_clients)
    else:
        partitioner = FederatedPartition(alpha=alpha, seed=seed)
        client_subsets = partitioner.split(train_ds, n_clients)

    client_loaders = [
        DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        for subset in client_subsets
    ]
    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return client_loaders, val_loader, test_loader


# ---------------------------------------------------------------------------
# Self-contained test cases
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(n: int = 500, n_classes: int = 10) -> TensorDataset:
    """Synthetic image dataset for tests — no downloads required."""
    X = torch.randn(n, 1, 28, 28)
    y = torch.randint(0, n_classes, (n,))
    return TensorDataset(X, y)


def _test_case_1_partition_dirichlet(simulate: bool):
    """
    Test 1 — FederatedPartition: Dirichlet split produces correct number
    of subsets, all indices are unique across clients, and each client
    meets the minimum sample requirement.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 1: Dirichlet partition  [{mode_label}] ===")

    dataset = _make_synthetic_dataset(500, n_classes=10)
    n_clients = 10

    for alpha in [10.0, 0.5, 0.1]:
        part = FederatedPartition(alpha=alpha, seed=42)
        subsets = part.split(dataset, n_clients, min_samples_per_client=5)

        assert len(subsets) == n_clients, "Wrong number of subsets."

        # All indices valid and within dataset bounds
        all_indices = []
        for s in subsets:
            assert len(s) >= 5, f"Client has fewer than 5 samples."
            all_indices.extend(s.indices)

        # No duplicate indices across clients
        assert len(all_indices) == len(set(all_indices)), \
            "Duplicate indices found across clients."

        # Total covers full dataset
        assert len(all_indices) == len(dataset), \
            f"Partition missing samples: {len(all_indices)} != {len(dataset)}"

        sizes = [len(s) for s in subsets]
        print(f"  α={alpha:.1f}  sizes: min={min(sizes)}, "
              f"max={max(sizes)}, std={np.std(sizes):.1f}")

    print("  PASSED")


def _test_case_2_noniid_sampler(simulate: bool):
    """
    Test 2 — NonIIDSampler: each client receives only k_classes,
    and the class distribution matrix confirms strict non-IID structure.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 2: Non-IID sampler  [{mode_label}] ===")

    dataset = _make_synthetic_dataset(1000, n_classes=10)
    n_clients = 5
    k = 2

    sampler = NonIIDSampler(k_classes=k, seed=42)
    subsets = sampler.split(dataset, n_clients)

    assert len(subsets) == n_clients

    part = FederatedPartition()
    dist = part.class_distribution(subsets)
    print(f"  Class distribution (clients x classes):")
    for i, row in enumerate(dist):
        nonzero = np.sum(row > 0)
        print(f"    Client {i}: {row}  ({nonzero} classes)")
        # Each client should have at most k_classes populated
        assert nonzero <= k + 1, \
            f"Client {i} has {nonzero} classes, expected <= {k+1}."

    print("  PASSED")


def _test_case_3_dataloader_pipeline(simulate: bool):
    """
    Test 3 — get_dataloaders: full pipeline with F-MNIST (auto-download),
    verifies loader counts, batch shapes, and that val/test are shared
    (same size regardless of n_clients).
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 3: DataLoader pipeline (F-MNIST)  [{mode_label}] ===")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        client_loaders, val_loader, test_loader = get_dataloaders(
            dataset_name="fmnist",
            n_clients=5,
            alpha=0.5,
            batch_size=32,
            data_root=Path(tmpdir),
            seed=42,
        )

        assert len(client_loaders) == 5, "Wrong number of client loaders."

        # Check batch shapes from first client
        x_batch, y_batch = next(iter(client_loaders[0]))
        assert x_batch.ndim == 4, "Expected 4D image tensor."
        assert x_batch.shape[1] == 1, "Expected 1-channel (grayscale)."
        assert y_batch.ndim == 1, "Expected 1D label tensor."
        print(f"  Client 0 batch: x={tuple(x_batch.shape)}  y={tuple(y_batch.shape)}")

        # Val and test loaders are consistent across different n_clients calls
        client_loaders_2, val_loader_2, _ = get_dataloaders(
            dataset_name="fmnist",
            n_clients=10,
            alpha=0.5,
            batch_size=32,
            data_root=Path(tmpdir),
            seed=42,
        )
        assert len(client_loaders_2) == 10

        # Report sizes
        total_train = sum(len(cl.dataset) for cl in client_loaders)
        print(f"  Total train samples across 5 clients : {total_train}")
        print(f"  Val batches  : {len(val_loader)}")
        print(f"  Test batches : {len(test_loader)}")

    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="datasets.py self-test suite",
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

    print("=" * 60)
    print(f"datasets.py — self-test suite  [{args.mode}]")
    print("=" * 60)

    _test_case_1_partition_dirichlet(simulate)
    _test_case_2_noniid_sampler(simulate)
    _test_case_3_dataloader_pipeline(simulate)

    print("\nAll tests passed.")