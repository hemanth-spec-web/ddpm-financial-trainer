"""
DDPM Training Loop
=====================
Trains the U-Net to predict noise given a noisy input and timestep.
Loss: MSE between predicted noise and actual noise added.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from app.services.ddpm_math import make_beta_schedule, precompute_schedule_values, q_sample
from app.services.unet import UNet1D
from app.services.financial_data import fetch_real_returns, build_training_windows, compute_stylized_facts


def generate_synthetic_dataset(n_samples: int = 1000, sequence_length: int = 256) -> torch.Tensor:
    """
    Generate a dataset of synthetic time-series (standing in for real
    financial data until we plug in yfinance in a later phase).
    Each sample is a sine-wave mixture with slightly randomized
    frequency/phase/noise, so the model has genuine variety to learn from.
    """
    torch.manual_seed(42)
    data = []
    for _ in range(n_samples):
        t_axis = torch.linspace(0, 4 * np.pi, sequence_length)
        freq1 = 1.0 + 0.3 * torch.rand(1).item()
        freq2 = 2.0 + 0.5 * torch.rand(1).item()
        phase = torch.rand(1).item() * 2 * np.pi

        signal = (
            0.5 * torch.sin(freq1 * t_axis + phase) +
            0.3 * torch.sin(freq2 * t_axis) +
            0.1 * torch.randn(sequence_length)
        )
        signal = (signal - signal.mean()) / (signal.std() + 1e-8)
        data.append(signal)

    return torch.stack(data)


def train_ddpm(
    T: int,
    beta_start: float,
    beta_end: float,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    sequence_length: int,
    d_model: int,
    data_source: str = "synthetic",
    ticker: str = "^GSPC",
    progress_callback=None,
) -> dict:
    """
    Full training loop. Returns final model state + loss history.

    progress_callback: optional function(epoch, total_epochs, loss) called
    after each epoch, used by the Celery task to report progress back to
    the database so the frontend can poll live status.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build noise schedule
    betas = make_beta_schedule(T=T, beta_start=beta_start, beta_end=beta_end)
    schedule = precompute_schedule_values(betas)
    schedule = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in schedule.items()}

    # Data — either synthetic sine-waves or real financial returns
    stylized_facts_real = None
    if data_source == "financial":
        real_returns = fetch_real_returns(ticker=ticker, period="5y")
        dataset = build_training_windows(real_returns, sequence_length=sequence_length, n_samples=1000)
        stylized_facts_real = compute_stylized_facts(dataset.numpy())
    else:
        dataset = generate_synthetic_dataset(n_samples=1000, sequence_length=sequence_length)

    dataloader = DataLoader(TensorDataset(dataset), batch_size=batch_size, shuffle=True)
    
    # Model
    model = UNet1D(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_losses = []

    for epoch in range(epochs):
        epoch_losses = []
        model.train()

        for (batch,) in dataloader:
            batch = batch.to(device)
            b_size = batch.shape[0]

            t = torch.randint(0, T, (b_size,), device=device)
            noise = torch.randn_like(batch)

            x_t, _ = q_sample(batch, t, schedule, noise=noise)

            predicted_noise = model(x_t, t)
            loss = loss_fn(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(round(avg_loss, 6))

        if progress_callback:
            progress_callback(epoch + 1, epochs, avg_loss)

    return {
        "train_losses": train_losses,
        "final_loss": train_losses[-1],
        "model_state": model.state_dict(),
        "stylized_facts_real": stylized_facts_real,
    }


@torch.no_grad()
def sample_from_model(
    model: UNet1D,
    schedule: dict,
    sequence_length: int,
    n_samples: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Reverse diffusion — generate new samples from pure noise.
    This is DDPM's actual generation process (Algorithm 2 in the paper).
    """
    model.eval()
    T = schedule["T"]

    x = torch.randn(n_samples, sequence_length, device=device)

    for t_step in reversed(range(T)):
        t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
        predicted_noise = model(x, t)

        alpha_t = schedule["alphas"][t_step]
        alpha_bar_t = schedule["alpha_bars"][t_step]
        beta_t = schedule["betas"][t_step]

        if t_step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha_t)) * (
            x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * noise

    return x