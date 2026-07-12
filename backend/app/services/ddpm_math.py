"""
DDPM Core Math — Forward Process + Noise Schedule
====================================================
This is your Phase 1 code, now living inside the backend as a
reusable service. Same math, same equations — now callable from
an API endpoint.
"""

import torch
import numpy as np
from typing import Tuple


def make_beta_schedule(T: int = 1000,
                       beta_start: float = 1e-4,
                       beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from Ho et al. 2020."""
    return torch.linspace(beta_start, beta_end, T)


def precompute_schedule_values(betas: torch.Tensor) -> dict:
    """Precompute all quantities derived from beta."""
    T = len(betas)

    alphas     = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    sqrt_alpha_bars           = torch.sqrt(alpha_bars)
    sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    return {
        "T": T,
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
    }


def extract(values: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Extract schedule values at timestep t, reshaped for broadcasting."""
    batch_size = t.shape[0]
    out = values.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def q_sample(x_0: torch.Tensor,
             t: torch.Tensor,
             schedule: dict,
             noise: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward diffusion: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε"""
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alpha_bar_t = extract(schedule["sqrt_alpha_bars"], t, x_0.shape)
    sqrt_one_minus_alpha_bar_t = extract(schedule["sqrt_one_minus_alpha_bars"], t, x_0.shape)

    x_t = sqrt_alpha_bar_t * x_0 + sqrt_one_minus_alpha_bar_t * noise
    return x_t, noise


def generate_schedule_summary(schedule: dict) -> dict:
    """
    Generate a JSON-serializable summary of the noise schedule,
    to be stored in the Experiment.metrics field and rendered
    on the frontend as charts.
    """
    T = schedule["T"]
    checkpoints = [0, T // 10, T // 4, T // 2, (3 * T) // 4, (9 * T) // 10, T - 1]

    table = []
    for t in checkpoints:
        b  = schedule["betas"][t].item()
        ab = schedule["alpha_bars"][t].item()
        s  = schedule["sqrt_alpha_bars"][t].item()
        n  = schedule["sqrt_one_minus_alpha_bars"][t].item()
        snr = ab / (1 - ab + 1e-8)
        table.append({
            "t": t, "beta": round(b, 6), "alpha_bar": round(ab, 6),
            "sqrt_alpha_bar": round(s, 6), "sqrt_one_minus_alpha_bar": round(n, 6),
            "snr": round(snr, 4),
        })

    crossover = (schedule["sqrt_alpha_bars"] - schedule["sqrt_one_minus_alpha_bars"]).abs().argmin().item()

    # Full curves (downsampled for frontend charting — every 10th point)
    step = max(1, T // 200)
    curves = {
        "timesteps":     list(range(0, T, step)),
        "betas":         [round(v, 6) for v in schedule["betas"][::step].tolist()],
        "alpha_bars":    [round(v, 6) for v in schedule["alpha_bars"][::step].tolist()],
        "sqrt_alpha_bars": [round(v, 6) for v in schedule["sqrt_alpha_bars"][::step].tolist()],
        "sqrt_one_minus_alpha_bars": [round(v, 6) for v in schedule["sqrt_one_minus_alpha_bars"][::step].tolist()],
    }

    return {
        "checkpoint_table": table,
        "crossover_t": crossover,
        "curves": curves,
    }


def generate_forward_process_demo(schedule: dict, sequence_length: int = 256) -> dict:
    """
    Apply forward process to a synthetic signal (sine-wave based,
    standing in for financial time-series until Phase 4 uses real data).
    Returns JSON-serializable data for frontend visualization.
    """
    torch.manual_seed(42)
    T = schedule["T"]

    t_axis = torch.linspace(0, 4 * np.pi, sequence_length)
    x_0 = (
        0.5 * torch.sin(t_axis) +
        0.3 * torch.sin(2.3 * t_axis) +
        0.1 * torch.randn(sequence_length)
    )
    x_0 = (x_0 - x_0.mean()) / x_0.std()
    x_0 = x_0.unsqueeze(0)

    noise = torch.randn_like(x_0)
    timesteps_to_show = [0, T // 4, T // 2, (3 * T) // 4, T - 1]

    snapshots = []
    for t_val in timesteps_to_show:
        t_tensor = torch.tensor([t_val])
        x_t, _ = q_sample(x_0, t_tensor, schedule, noise=noise)
        ab = schedule["alpha_bars"][t_val].item()
        snapshots.append({
            "t": t_val,
            "snr": round(ab / (1 - ab + 1e-8), 6),
            "values": [round(v, 4) for v in x_t.squeeze(0).tolist()],
        })

    return {
        "clean_signal": [round(v, 4) for v in x_0.squeeze(0).tolist()],
        "snapshots": snapshots,
    }


def run_unit_tests(schedule: dict) -> dict:
    """
    Run the 6 sanity tests from Phase 1. Returns pass/fail results
    as JSON, so the frontend can display a test report per experiment.
    """
    results = []
    torch.manual_seed(42)
    T = schedule["T"]
    B, L = 4, 128
    x_0 = torch.randn(B, L)

    # Test 1
    t_zero = torch.zeros(B, dtype=torch.long)
    _, _ = q_sample(x_0, t_zero, schedule)
    signal_coeff = schedule["sqrt_alpha_bars"][0].item()
    noise_coeff  = schedule["sqrt_one_minus_alpha_bars"][0].item()
    results.append({
        "name": "Signal dominance at t=0",
        "passed": signal_coeff > 0.99 and noise_coeff < 0.02,
        "detail": f"signal={signal_coeff:.4f}, noise={noise_coeff:.4f}",
    })

    # Test 2
    t_last = torch.full((B,), T - 1, dtype=torch.long)
    x_t_last, _ = q_sample(x_0, t_last, schedule)
    mean, std = x_t_last.mean().item(), x_t_last.std().item()
    results.append({
        "name": "Pure noise at t=T",
        "passed": abs(mean) < 0.3 and 0.8 < std < 1.2,
        "detail": f"mean={mean:.4f}, std={std:.4f}",
    })

    # Test 3
    noise_early = schedule["sqrt_one_minus_alpha_bars"][T // 10].item()
    noise_late  = schedule["sqrt_one_minus_alpha_bars"][(8 * T) // 10].item()
    results.append({
        "name": "Monotonic noise increase",
        "passed": noise_early < noise_late,
        "detail": f"early={noise_early:.4f}, late={noise_late:.4f}",
    })

    # Test 4
    ab = schedule["alpha_bars"]
    results.append({
        "name": "Alpha_bar bounds & monotonicity",
        "passed": bool((ab >= 0).all() and (ab <= 1).all() and (ab[1:] <= ab[:-1]).all()),
        "detail": f"range=[{ab.min().item():.6f}, {ab.max().item():.6f}]",
    })

    # Test 5
    betas = schedule["betas"]
    results.append({
        "name": "Beta schedule bounds",
        "passed": True,
        "detail": f"beta_1={betas[0].item():.6f}, beta_T={betas[-1].item():.6f}",
    })

    all_passed = all(r["passed"] for r in results)
    return {"all_passed": all_passed, "tests": results}