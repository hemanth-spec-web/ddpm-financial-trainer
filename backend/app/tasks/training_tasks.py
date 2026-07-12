"""
Celery Task — Background DDPM Training
==========================================
Runs in a separate worker process so the API stays responsive
during long training runs (minutes to hours).

Uses a SYNCHRONOUS SQLAlchemy session because Celery workers don't
share the FastAPI event loop — this is standard practice for
mixing Celery with an async web framework.
"""

import os
import torch
from datetime import datetime, timezone
from sqlalchemy import select

from app.tasks.celery_app import celery_app
from app.tasks.db import SyncSession
from app.models.experiment import Experiment, ExperimentStatus
from app.services.training import train_ddpm

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model_weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)


@celery_app.task(bind=True, name="train_ddpm_task")
def train_ddpm_task(self, experiment_id: str):
    """
    Background task: trains a DDPM model for the given experiment
    and updates its row in the database with live progress + final results.
    """
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
            """Called after every epoch — updates DB so frontend can poll."""
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

        # Save model weights to disk
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