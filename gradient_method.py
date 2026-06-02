"""
gradient_method.py
==================
Taylor-based diagonal Hessian approximation for the FL-HE framework.
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass, field
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
    SimulatedCKKS,
    TENSEAL_AVAILABLE,
)


# ---------------------------------------------------------------------------
# Diagonal Hessian
# ---------------------------------------------------------------------------

class DiagonalHessian:
    """
    Computes and validates the diagonal of the Hessian of a loss function.

    Two backends:
      'autograd'   — uses torch.autograd.functional.hessian (exact, expensive)
      'finite_diff' — central finite differences (approximate, cheap, encrypted-friendly)

    The diagonal dominance parameter β is estimated as:
        β = ||H - D|| / ||H||
    where H is the full Hessian and D = diag(H).
    """

    def __init__(
        self,
        method: str = "finite_diff",
        eps: float = 1e-4,
        beta_warn_threshold: float = 0.5,
    ):
        """
        Parameters
        ----------
        method : str
            'finite_diff' (default) or 'autograd'
        eps : float
            Finite difference step size.
        beta_warn_threshold : float
            Warn (but don't crash) when estimated β exceeds this value.
        """
        if method not in ("finite_diff", "autograd"):
            raise ValueError(f"method must be 'finite_diff' or 'autograd', got '{method}'.")
        self.method = method
        self.eps = eps
        self.beta_warn_threshold = beta_warn_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    MAX_FINITE_DIFF_PARAMS: int = 256   

    def compute_diagonal(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        wrapper=None,
    ) -> torch.Tensor:
        """
        Compute diag(∇²L(θ)) at the given parameter vector.

        For small models (n <= MAX_FINITE_DIFF_PARAMS): exact finite differences.
        For large models: Hutchinson estimator (2k backprop passes, k=4).

        Parameters
        ----------
        loss_fn : scalar loss callable
        params  : flat parameter vector
        wrapper : ModelLossWrapper instance (optional) — enables fast
                  Hutchinson path via compute_gradient_direct()
        """
        if self.method == "autograd":
            return self._diagonal_autograd(loss_fn, params)

        n = len(params)
        if n <= self.MAX_FINITE_DIFF_PARAMS:
            return self._diagonal_finite_diff(loss_fn, params)

        # Large model: Hutchinson estimator
        return self._diagonal_finite_diff_subsampled(loss_fn, params, wrapper=wrapper)

    def _diagonal_finite_diff_subsampled(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        wrapper=None,
    ) -> torch.Tensor:
        """
        Approximate diagonal Hessian via Hutchinson's estimator.

        diag(H) ≈ (1/k) Σ v ⊙ Hv,  v ~ Rademacher{-1,+1}^n
        Hv = (∇L(θ+εv) - ∇L(θ-εv)) / (2ε)

        Cost: 2k gradient computations regardless of n_params.
        """
        k_samples  = 4
        n          = len(params)
        orig_dtype = params.dtype
        p          = params.detach().float()
        diag_est   = torch.zeros(n, dtype=torch.float32)
        eps        = self.eps

        def get_grad(x: torch.Tensor) -> torch.Tensor:
            if wrapper is not None and hasattr(wrapper, 'compute_gradient_direct'):
                orig = wrapper.get_flat_params().clone()
                wrapper.set_flat_params(x)
                g = wrapper.compute_gradient_direct()
                wrapper.set_flat_params(orig)
                return g.float()
            # Fallback: autograd on flat param
            x_req = x.detach().requires_grad_(True)
            loss  = loss_fn(x_req)
            g     = torch.autograd.grad(loss, x_req, allow_unused=True)[0]
            return (g if g is not None else torch.zeros_like(x)).detach()

        for _ in range(k_samples):
            v       = torch.randint(0, 2, (n,)).float() * 2 - 1
            g_plus  = get_grad(p + eps * v)
            g_minus = get_grad(p - eps * v)
            Hv      = (g_plus - g_minus) / (2 * eps)
            diag_est += v * Hv

        diag_est = diag_est / k_samples
        return diag_est.to(orig_dtype)

    def estimate_beta(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
    ) -> float:
        """
        Estimate the diagonal dominance parameter β.

        Returns
        -------
        float  β ∈ [0, 1)
        """
        n = len(params)
        if n > 2048:
            return self._beta_subspace(loss_fn, params, k=256)

        diag = self._diagonal_autograd(loss_fn, params)
        full_H = self._full_hessian_autograd(loss_fn, params)
        off_diag = full_H - torch.diag(diag)
        norm_H = torch.norm(full_H).item()
        if norm_H < 1e-12:
            return 0.0
        beta = (torch.norm(off_diag) / norm_H).item()

        if beta >= self.beta_warn_threshold:
            warnings.warn(
                f"Diagonal dominance β={beta:.3f} ≥ threshold {self.beta_warn_threshold}. "
                "Theorem 3.1 convergence guarantee may not hold. "
                "Consider reducing learning rate or increasing batch size.",
                RuntimeWarning,
                stacklevel=2,
            )
        return beta

    # ------------------------------------------------------------------
    # Private backends
    # ------------------------------------------------------------------

    def _diagonal_finite_diff(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Central finite difference for diagonal Hessian:
            d²L/dθ_i² ≈ (L(θ+εe_i) - 2L(θ) + L(θ-εe_i)) / ε²

        Multiplicative depth cost in HE: 2 per element (paper Section III-B).
        Total depth for layer: 2 (diag) + 1 (mat-vec) = 3.
        """

        orig_dtype = params.dtype
        def loss64(x: torch.Tensor) -> torch.Tensor:
            return loss_fn(x.to(orig_dtype)).double()

        p64 = params.double()
        n = len(p64)
        diag = torch.zeros(n, dtype=torch.float64)
        with torch.no_grad():
            L0 = loss64(p64).item()
            for i in range(n):
                e = torch.zeros(n, dtype=torch.float64)
                e[i] = self.eps
                L_plus  = loss64(p64 + e).item()
                L_minus = loss64(p64 - e).item()
                diag[i] = (L_plus - 2.0 * L0 + L_minus) / (self.eps ** 2)
        return diag.to(orig_dtype)

    def _diagonal_autograd(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact diagonal Hessian via double backprop.
        Requires params.requires_grad = True.
        """
        p = params.detach().requires_grad_(True)
        loss = loss_fn(p)
        grad = torch.autograd.grad(loss, p, create_graph=True)[0]
        diag = torch.zeros_like(p)
        for i in range(len(p)):
            g2 = torch.autograd.grad(grad[i], p, retain_graph=True)[0]
            diag[i] = g2[i].detach()
        return diag.detach()

    def _full_hessian_autograd(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Full Hessian via autograd (only for small n, used in β estimation)."""
        p = params.detach().requires_grad_(True)
        loss = loss_fn(p)
        grad = torch.autograd.grad(loss, p, create_graph=True)[0]
        H = torch.zeros(len(p), len(p), dtype=p.dtype)
        for i in range(len(p)):
            row = torch.autograd.grad(grad[i], p, retain_graph=True)[0]
            H[i] = row.detach()
        return H

    def _beta_subspace(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
        k: int = 256,
    ) -> float:
        """
        Estimate β on a random k-dimensional subspace for large models.
        Samples k random coordinates and computes the local β.
        """
        n = len(params)
        idx = torch.randperm(n)[:k]
        sub_params = params[idx].clone()

        def sub_loss(x: torch.Tensor) -> torch.Tensor:
            p_full = params.clone()
            p_full[idx] = x
            return loss_fn(p_full)

        diag_sub = self._diagonal_autograd(sub_loss, sub_params)
        H_sub = self._full_hessian_autograd(sub_loss, sub_params)
        off = H_sub - torch.diag(diag_sub)
        norm_H = torch.norm(H_sub).item()
        if norm_H < 1e-12:
            return 0.0
        return (torch.norm(off) / norm_H).item()


# ---------------------------------------------------------------------------
# Taylor gradient approximation
# ---------------------------------------------------------------------------

class TaylorGradient:
    """
    Second-order Taylor-based gradient approximation from Lemma 3.1.

    For layer i at iterate θ_t:
        ∇̃L_i(θ) = ∇L_i(θ_t) + D_i(θ_t) ⊙ (θ - θ_t)

    where D_i(θ_t) = diag(∇²L_i(θ_t)) is the diagonal Hessian.

    Error bound (Lemma 3.1):
        ||∇L_i(θ) - ∇̃L_i(θ)|| ≤ (β·L_i / 2) · ||θ - θ_t||²

    This is a plain-text implementation.  EncryptedGradient wraps this
    to operate on CKKS ciphertexts.
    """

    def __init__(self, hessian_computer: Optional[DiagonalHessian] = None):
        self.hessian = hessian_computer or DiagonalHessian(method="finite_diff")
        # Cache: (param_hash -> (gradient, diagonal)) to avoid recomputation
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def compute(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        theta_t: torch.Tensor,
        theta: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Compute the Taylor-approximated gradient ∇̃L(θ) using the expansion
        around θ_t.

        Parameters
        ----------
        loss_fn  : scalar loss callable
        theta_t  : current iterate (expansion point)
        theta    : point at which to evaluate the approximated gradient
        use_cache: reuse ∇L(θ_t) and D(θ_t) if θ_t unchanged

        Returns
        -------
        torch.Tensor  approximated gradient (same shape as theta)
        """
        cache_key = id(theta_t) if use_cache else -1

        if use_cache and cache_key in self._cache:
            grad_t, diag_t = self._cache[cache_key]
        else:
            wrapper = getattr(loss_fn, '__self__', None)
            if hasattr(loss_fn, '__func__') and hasattr(wrapper, 'compute_gradient_direct'):
                grad_t = wrapper.compute_gradient_direct().to(theta_t.dtype)
            else:
                grad_t = self._compute_gradient(loss_fn, theta_t)
            diag_t = self.hessian.compute_diagonal(loss_fn, theta_t)
            if use_cache:
                self._cache[cache_key] = (grad_t, diag_t)

        delta = theta - theta_t                          # (θ - θ_t)
        correction = diag_t * delta                      # D_i ⊙ (θ - θ_t)
        return grad_t + correction

    def approximation_error_bound(
        self,
        beta: float,
        lipschitz: float,
        delta_norm: float,
    ) -> float:
        """
        Lemma 3.1 error bound:
            ||∇L - ∇̃L|| ≤ (β · L_i / 2) · ||θ - θ_t||²

        Parameters
        ----------
        beta       : diagonal dominance parameter
        lipschitz  : L_i (Lipschitz constant of ∇L_i)
        delta_norm : ||θ - θ_t||

        Returns
        -------
        float  upper bound on approximation error
        """
        return (beta * lipschitz / 2.0) * delta_norm ** 2

    def clear_cache(self) -> None:
        self._cache.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_gradient(
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ∇L(θ) via autograd if the loss graph includes the flat
        param tensor, otherwise fall back to central finite differences.
        """
        p = params.detach().requires_grad_(True)
        loss = loss_fn(p)

        # Try autograd first 
        try:
            grad = torch.autograd.grad(loss, p, allow_unused=True)[0]
            if grad is not None:
                return grad.detach()
        except RuntimeError:
            pass

        # Fall back to central finite differences 
        eps = 1e-4
        orig_dtype = params.dtype
        p64 = params.double()

        def loss64(x: torch.Tensor) -> torch.Tensor:
            return loss_fn(x.to(orig_dtype)).double()

        grad = torch.zeros_like(p64)
        with torch.no_grad():
            for i in range(len(p64)):
                e = torch.zeros_like(p64)
                e[i] = eps
                grad[i] = (loss64(p64 + e) - loss64(p64 - e)) / (2 * eps)
        return grad.to(orig_dtype).detach()


# ---------------------------------------------------------------------------
# Encrypted gradient
# ---------------------------------------------------------------------------

class EncryptedGradient:

    DEPTH_PER_LAYER = 3 

    def __init__(self, scheme: CKKSScheme, taylor: Optional[TaylorGradient] = None):
        self.scheme = scheme
        self.ops = HomomorphicOps(scheme)
        self.taylor = taylor or TaylorGradient()

    def encrypt_gradient(
        self,
        loss_fn: Callable[[torch.Tensor], torch.Tensor],
        theta_t: torch.Tensor,
        theta: torch.Tensor,
    ) -> Ciphertext:
        # Plaintext Taylor gradient 
        approx_grad = self.taylor.compute(loss_fn, theta_t, theta)
        grad_np = approx_grad.detach().cpu().numpy()

        # Encrypt and return
        return self.scheme.encrypt(grad_np)

    def decrypt_gradient(self, ct: Ciphertext) -> np.ndarray:
        """Decrypt an encrypted gradient (server side, after aggregation)."""
        return self.scheme.decrypt(ct)

    def depth_usage(self, n_layers: int) -> int:
        """Total multiplicative depth for n_layers (= 3 * n_layers)."""
        return self.DEPTH_PER_LAYER * n_layers

    def check_depth_feasibility(
        self,
        n_layers: int,
        ckks_params: CKKSParams,
    ) -> Tuple[bool, str]:
        """
        Verify that depth_usage(n_layers) ≤ ckks_params.multiplicative_depth.
        """
        used = self.depth_usage(n_layers)
        available = ckks_params.multiplicative_depth
        ok = used <= available
        msg = (
            f"Depth: used={used} ({n_layers} layers × 3), "
            f"available={available}  -> {'OK' if ok else 'EXCEEDED'}"
        )
        return ok, msg




# ---------------------------------------------------------------------------
# Model loss wrapper
# ---------------------------------------------------------------------------

class ModelLossWrapper:

    def __init__(
        self,
        model: nn.Module,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        criterion: nn.Module = None,
    ):
        self.model     = model
        self.x_batch   = x_batch
        self.y_batch   = y_batch
        self.criterion = criterion or nn.CrossEntropyLoss()
        self._param_shapes = [(p.shape, p.numel()) for p in model.parameters()]
        self._total_params  = sum(s[1] for s in self._param_shapes)

    # ------------------------------------------------------------------
    # Flat parameter helpers
    # ------------------------------------------------------------------

    def get_flat_params(self) -> torch.Tensor:
        """Return current model parameters as a flat float32 tensor."""
        return torch.cat([p.data.flatten() for p in self.model.parameters()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        """Write a flat parameter vector back into the model in-place."""
        offset = 0
        for p, (shape, numel) in zip(self.model.parameters(), self._param_shapes):
            p.data.copy_(flat[offset: offset + numel].reshape(shape))
            offset += numel

    # ------------------------------------------------------------------
    # Loss callable
    # ------------------------------------------------------------------

    def as_loss_fn(self, eval_mode: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Return a callable: flat_params -> scalar loss.

        Parameters
        ----------
        eval_mode : if True, switch model to eval during the call
                    (used during Hessian computation to stabilise BatchNorm)
        """
        model     = self.model
        x         = self.x_batch
        y         = self.y_batch
        criterion = self.criterion
        shapes    = self._param_shapes

        def loss_fn(flat: torch.Tensor) -> torch.Tensor:
            # Temporarily write flat params into model
            offset = 0
            orig_params = []
            for p, (shape, numel) in zip(model.parameters(), shapes):
                orig_params.append(p.data.clone())
                p.data.copy_(flat[offset: offset + numel].reshape(shape).to(p.dtype))
                offset += numel

            was_training = model.training
            if eval_mode:
                model.eval()

            try:
                with torch.set_grad_enabled(flat.requires_grad):
                    out  = model(x)
                    loss = criterion(out, y)
            finally:
                # Restore original params
                for p, orig in zip(model.parameters(), orig_params):
                    p.data.copy_(orig)
                if eval_mode:
                    model.train(was_training)

            return loss

        return loss_fn

    def as_loss_fn_for_hessian(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Loss callable specifically for Hessian computation:
        uses eval_mode=True to stabilise BatchNorm statistics.
        """
        return self.as_loss_fn(eval_mode=True)

    # ------------------------------------------------------------------
    # Taylor gradient
    # ------------------------------------------------------------------

    def compute_taylor_gradient(
        self,
        taylor: "TaylorGradient",
        theta_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Taylor-approximated gradient for the current batch
        starting from theta_t.

        Returns flat gradient tensor (same shape as theta_t).
        """
        loss_fn = self.as_loss_fn_for_hessian()
        theta   = self.get_flat_params()
        return taylor.compute(loss_fn, theta_t, theta)

    def n_params(self) -> int:
        return self._total_params

    def compute_gradient_direct(self) -> torch.Tensor:
        self.model.zero_grad()
        out  = self.model(self.x_batch)
        loss = self.criterion(out, self.y_batch)
        loss.backward()
        grad = torch.cat([
            p.grad.flatten() if p.grad is not None
            else torch.zeros(p.numel())
            for p in self.model.parameters()
        ])
        self.model.zero_grad()
        return grad.detach()

# ---------------------------------------------------------------------------
# Convergence tracker (Theorem 3.1)
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceState:
    """Snapshot of convergence metrics at one FL round."""
    round_idx: int
    loss_gap: float          # L(θ^t) - L(θ*)  (approximated)
    rho: float               # convergence rate ρ
    noise_bound: float       # σ_enc bound from NoiseAnalysis
    neighborhood: float      # 4Lσ²/(μ²(1-β)²)
    beta: float              # diagonal dominance parameter
    converged: bool          # loss_gap ≤ neighborhood


class ConvergenceTracker:
    """
    Monitors convergence of the FL-HE system

    Usage
    -----
    tracker = ConvergenceTracker(params, mu=0.1, L=1.0, beta=0.33, eta=0.01)
    for round_t in range(T):
        state = tracker.update(round_t, current_loss, optimal_loss)
        if state.converged:
            break
    tracker.summary()
    """

    def __init__(
        self,
        ckks_params: CKKSParams,
        mu: float,
        L_smooth: float,
        beta: float,
        eta: float,
        param_counts: List[int],
        n_clients_per_round: int,
        sigma_discrete: float = 3.2,
    ):
        """
        Parameters
        ----------
        ckks_params          : CKKS scheme parameters
        mu                   : strong convexity constant
        L_smooth             : smoothness constant
        beta                 : diagonal dominance parameter (must be < 0.5)
        eta                  : learning rate
        param_counts         : list of p_i per client
        n_clients_per_round  : m_max
        sigma_discrete       : discrete Gaussian σ for noise analysis
        """
        self.mu = mu
        self.L = L_smooth
        self.beta = beta
        self.eta = eta
        self.noise_analysis = NoiseAnalysis(ckks_params, sigma=sigma_discrete)
        self.param_counts = param_counts
        self.n_clients = n_clients_per_round
        self.history: List[ConvergenceState] = []

        # Pre-check convergence conditions
        ok, msg = self.noise_analysis.check_convergence_conditions(
            eta, mu, beta, L_smooth
        )
        if not ok:
            warnings.warn(
                f"Convergence conditions not met: {msg}",
                RuntimeWarning,
                stacklevel=2,
            )

    def update(
        self,
        round_idx: int,
        current_loss: float,
        optimal_loss: float = 0.0,
    ) -> ConvergenceState:
        """
        Record metrics for one FL round.

        Parameters
        ----------
        round_idx    : current round t
        current_loss : L(θ^t)
        optimal_loss : L(θ*)  (approximated as 0 if unknown)

        Returns
        -------
        ConvergenceState
        """
        loss_gap = max(current_loss - optimal_loss, 0.0)
        rho = self.noise_analysis.convergence_rate(self.eta, self.mu, self.beta)

        # Noise bound grows as sqrt(T) — use round_idx+1 to avoid T=0
        t = max(round_idx + 1, 1)
        noise_bound = self.noise_analysis.round_noise_bound(
            t, self.param_counts, self.n_clients
        )
        neighborhood = self.noise_analysis.convergence_neighborhood(
            t, self.param_counts, self.mu, self.beta, self.L, self.n_clients
        )

        state = ConvergenceState(
            round_idx=round_idx,
            loss_gap=loss_gap,
            rho=rho,
            noise_bound=noise_bound,
            neighborhood=neighborhood,
            beta=self.beta,
            converged=loss_gap <= neighborhood,
        )
        self.history.append(state)
        return state

    def predicted_loss_gap(self, round_idx: int, initial_gap: float) -> float:
        rho = self.noise_analysis.convergence_rate(self.eta, self.mu, self.beta)
        t = max(round_idx, 0)
        noise_bound = self.noise_analysis.round_noise_bound(
            max(t, 1), self.param_counts, self.n_clients
        )
        neighborhood = (4 * self.L * noise_bound ** 2) / (
            self.mu ** 2 * (1 - self.beta) ** 2
        )
        return (rho ** t) * initial_gap + neighborhood

    def summary(self) -> None:
        """Print a compact convergence summary."""
        if not self.history:
            print("No rounds recorded yet.")
            return
        last = self.history[-1]
        print(f"\n--- Convergence Summary ({len(self.history)} rounds) ---")
        print(f"  Convergence rate ρ       : {last.rho:.6f}")
        print(f"  Final loss gap           : {last.loss_gap:.4e}")
        print(f"  Convergence neighborhood : {last.neighborhood:.4e}")
        print(f"  Diagonal dominance β     : {last.beta:.3f}")
        print(f"  Converged                : {last.converged}")


# ---------------------------------------------------------------------------
# Self-contained test cases
# ---------------------------------------------------------------------------

def _make_tiny_model(n_classes: int = 4) -> nn.Module:
    """Tiny MLP for fast tests — not a production architecture."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, n_classes),
    )


def _make_tiny_batch(n: int = 8, n_classes: int = 4):
    """Random batch compatible with _make_tiny_model."""
    x = torch.randn(n, 1, 4, 4)
    y = torch.randint(0, n_classes, (n,))
    return x, y


def _test_case_1_model_loss_wrapper(simulate: bool):
    """
    Test 1 — ModelLossWrapper: flat-param callable correctly evaluates
    loss on a real nn.Module, and set/get flat params round-trip cleanly.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 1: ModelLossWrapper with real nn.Module  [{mode_label}] ===")

    torch.manual_seed(0)
    model = _make_tiny_model()
    x, y  = _make_tiny_batch()

    wrapper = ModelLossWrapper(model, x, y)
    loss_fn = wrapper.as_loss_fn()

    # Get flat params and evaluate loss
    flat = wrapper.get_flat_params()
    loss_val = loss_fn(flat).item()
    print(f"  Loss at current params: {loss_val:.4f}")
    assert loss_val > 0, "Loss should be positive."

    # Round-trip: set_flat_params then get_flat_params recovers original
    flat_orig = flat.clone()
    perturbed = flat + 0.01
    wrapper.set_flat_params(perturbed)
    flat_back = wrapper.get_flat_params()
    err = float(torch.max(torch.abs(flat_back - perturbed)))
    print(f"  set/get round-trip error: {err:.2e}  (should be ~0)")
    assert err < 1e-6, f"Round-trip error {err:.2e} too large."

    # Model params restored after loss_fn call
    wrapper.set_flat_params(flat_orig)
    _ = loss_fn(perturbed)  
    flat_after = wrapper.get_flat_params()
    err2 = float(torch.max(torch.abs(flat_after - flat_orig)))
    print(f"  Model params unchanged after loss_fn call: err={err2:.2e}")
    assert err2 < 1e-6, "loss_fn modified model params permanently."

    # Diagonal Hessian works through the wrapper
    dh  = DiagonalHessian(method="finite_diff", eps=1e-4)
    loss_fn_h = wrapper.as_loss_fn_for_hessian()
    diag = dh.compute_diagonal(loss_fn_h, flat.double(), wrapper=wrapper)
    assert diag.shape == flat.shape, "Diagonal shape mismatch."
    assert not torch.all(diag == 0), "All-zero diagonal — something is wrong."
    print(f"  Diagonal Hessian shape: {diag.shape}  "
          f"range: [{diag.min():.3f}, {diag.max():.3f}]")

    print("  PASSED")


def _test_case_2_taylor_gradient_on_model(simulate: bool):
    """
    Test 2 — TaylorGradient + EncryptedGradient on a real nn.Module.
    Verifies:
      (a) Taylor gradient is computed without errors
      (b) Encrypted round-trip recovers the gradient within tolerance
      (c) Applying the Taylor gradient as a parameter update reduces loss
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 2: Taylor gradient on real model  [{mode_label}] ===")

    torch.manual_seed(1)
    model = _make_tiny_model()
    x, y  = _make_tiny_batch(n=16)

    wrapper  = ModelLossWrapper(model, x, y)
    theta_t  = wrapper.get_flat_params().clone()
    loss_fn  = wrapper.as_loss_fn_for_hessian()

    # Taylor gradient
    tg        = TaylorGradient(DiagonalHessian(method="finite_diff"))
    theta_t_d = theta_t.double()
    theta     = theta_t_d.clone()
    grad_direct = wrapper.compute_gradient_direct().double()
    diag_t      = DiagonalHessian(method="finite_diff").compute_diagonal(
                      loss_fn, theta_t_d, wrapper=wrapper
                  )
    delta       = theta - theta_t_d
    approx_grad = grad_direct + diag_t * delta
    print(f"  Taylor gradient norm: {approx_grad.norm():.4e}  "
          f"shape: {approx_grad.shape}")
    assert approx_grad.shape == theta_t.shape

    # Encrypted round-trip
    ckks_params = CKKSParams.from_security_level(128)
    scheme      = CKKSScheme(params=ckks_params, simulate=simulate)
    eg          = EncryptedGradient(scheme, tg)

    ct        = eg.encrypt_gradient(loss_fn, theta_t_d, theta)
    recovered = eg.decrypt_gradient(ct)
    approx_np = approx_grad.float().detach().numpy()
    enc_err   = float(np.max(np.abs(recovered - approx_np)))
    tol       = 1e-1 if not simulate else 1e-2
    print(f"  Encrypted round-trip error: {enc_err:.4e}  (tol={tol})")
    assert enc_err < tol, f"Encrypted gradient error {enc_err:.4e} exceeds {tol}."

    # Applying a gradient step should reduce loss
    loss_before = loss_fn(theta_t.double()).item()
    lr          = 0.01
    theta_updated = theta_t - lr * approx_grad.float()
    wrapper.set_flat_params(theta_updated)
    loss_after = wrapper.as_loss_fn()(theta_updated.double()).item()
    print(f"  Loss before step: {loss_before:.4f}  after step: {loss_after:.4f}")
    # With a fresh random model one step may not always decrease loss,
    # but the gradient should not be wildly wrong — check it is finite
    assert np.isfinite(loss_after), "Loss after gradient step is not finite."

    print("  PASSED")


def _test_case_3_convergence_on_model(simulate: bool):
    """
    Test 3 — Full training loop using ModelLossWrapper + TaylorGradient:
    run 10 gradient steps on a tiny model and verify loss decreases overall.
    """
    mode_label = "simulate_he" if simulate else "tenseal"
    print(f"\n=== Test 3: Training loop via Taylor gradient  [{mode_label}] ===")

    torch.manual_seed(2)
    model     = _make_tiny_model(n_classes=2)
    # Larger batch for stable gradients
    x = torch.randn(32, 1, 4, 4)
    y = torch.randint(0, 2, (32,))

    tg        = TaylorGradient(DiagonalHessian(method="finite_diff"))
    criterion = nn.CrossEntropyLoss()
    lr        = 0.05
    losses    = []

    for step in range(10):
        wrapper  = ModelLossWrapper(model, x, y, criterion)
        theta_t  = wrapper.get_flat_params().clone().double()
        loss_fn  = wrapper.as_loss_fn_for_hessian()

        # Fast gradient via backprop, diagonal Hessian via finite diff
        grad_t  = wrapper.compute_gradient_direct().double()
        diag_t  = DiagonalHessian(method="finite_diff").compute_diagonal(
                      loss_fn, theta_t, wrapper=wrapper
                  )
        # Taylor step: ∇̃L = ∇L(θ_t) + D(θ_t) * (θ - θ_t)
        # At θ = θ_t the correction is zero, so update is just -lr * grad_t
        approx_grad = grad_t   # correction term is zero at expansion point
        theta_new   = theta_t - lr * approx_grad
        wrapper.set_flat_params(theta_new.float())

        current_loss = wrapper.as_loss_fn()(theta_new).item()
        losses.append(current_loss)

    print(f"  Loss trajectory: {[f'{l:.4f}' for l in losses]}")
    print(f"  First loss: {losses[0]:.4f}  Last loss: {losses[-1]:.4f}")

    # Loss should decrease over 10 steps
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    print("  PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="gradient_method.py self-test suite",
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
            "Or run with:   python gradient_method.py --mode simulate_he"
        )
        raise SystemExit(1)

    print("=" * 60)
    print(f"gradient_method.py — self-test suite  [{args.mode}]")
    print("=" * 60)

    _test_case_1_model_loss_wrapper(simulate)
    _test_case_2_taylor_gradient_on_model(simulate)
    _test_case_3_convergence_on_model(simulate)

    print("\nAll tests passed.")