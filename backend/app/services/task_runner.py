"""
Shared task logic — used by BOTH the Celery task wrappers (local dev)
and the direct background-task fallback (free-tier deployment without
a Celery worker). Keeping the logic here means both paths behave
identically; only how they get invoked differs.
"""

import os
import torch
from datetime import datetime, timezone
from sqlalchemy import select

from app.tasks.db import SyncSession
from app.models.experiment import Experiment, ExperimentStatus
from app.services.training import train_ddpm, sample_from_model
from app.services.unet import UNet1D
from app.services.ddpm_math import make_beta_schedule, precompute_schedule_values
from app.services.financial_data import compute_stylized_facts

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


def execute_training(experiment_id: str):
    """Runs the full training loop and updates the DB. Safe to call
    directly (no Celery) or from within a Celery task."""
    session = SyncSession()

    try:
        experiment = session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()

        if not experiment:
            return {"error": "Experiment not found"}

        experiment.status = ExperimentStatus.RUNNING
        session.commit()

        def progress_callback(epoch, total_epochs, loss):
            experiment.current_epoch = epoch
            losses = list(experiment.train_losses) if experiment.train_losses else []
            losses.append(round(loss, 6))
            experiment.train_losses = losses
            session.commit()

        result = train_ddpm(
            T=experiment.T,
            beta_start=experiment.beta_start,
            beta_end=experiment.beta_end,
            epochs=experiment.epochs,
            batch_size=experiment.batch_size,
            learning_rate=experiment.learning_rate,
            sequence_length=experiment.sequence_length,
            d_model=experiment.d_model,
            data_source=experiment.data_source,
            ticker=experiment.ticker,
            progress_callback=progress_callback,
        )

        weights_path = os.path.join(WEIGHTS_DIR, f"{experiment_id}.pt")
        torch.save(result["model_state"], weights_path)

        experiment.model_weights_path = weights_path
        experiment.final_loss = result["final_loss"]

        if result.get("stylized_facts_real"):
            current_metrics = dict(experiment.metrics) if experiment.metrics else {}
            current_metrics["stylized_facts_real_data"] = result["stylized_facts_real"]
            experiment.metrics = current_metrics

        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)
        session.commit()

        return {"status": "completed", "final_loss": result["final_loss"]}

    except Exception as e:
        session.rollback()
        experiment = session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()
        if experiment:
            experiment.status = ExperimentStatus.FAILED
            experiment.metrics = {**(experiment.metrics or {}), "training_error": str(e)}
            session.commit()
        raise

    finally:
        session.close()


def execute_generation(experiment_id: str, n_samples: int = 4):
    """Runs reverse diffusion sampling and updates the DB. Safe to call
    directly (no Celery) or from within a Celery task."""
    session = SyncSession()

    try:
        experiment = session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()

        if not experiment or not experiment.model_weights_path:
            return {"error": "No trained model found for this experiment"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = UNet1D(d_model=experiment.d_model).to(device)
        state_dict = torch.load(experiment.model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        betas = make_beta_schedule(
            T=experiment.T, beta_start=experiment.beta_start, beta_end=experiment.beta_end
        )
        schedule = precompute_schedule_values(betas)
        schedule = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in schedule.items()}

        generated = sample_from_model(
            model=model,
            schedule=schedule,
            sequence_length=experiment.sequence_length,
            n_samples=n_samples,
            device=device,
        )

        generated_np = generated.cpu().numpy()
        stylized_facts_generated = compute_stylized_facts(generated_np)

        current_samples = dict(experiment.generated_samples) if experiment.generated_samples else {}
        current_samples["model_generated"] = {
            "sequences": [[round(v, 4) for v in seq.tolist()] for seq in generated.cpu()]
        }
        experiment.generated_samples = current_samples

        current_metrics = dict(experiment.metrics) if experiment.metrics else {}
        current_metrics["stylized_facts_generated"] = stylized_facts_generated
        experiment.metrics = current_metrics

        session.commit()
        return {"status": "completed", "n_samples": n_samples}

    except Exception as e:
        session.rollback()
        raise

    finally:
        session.close()