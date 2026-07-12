"""
Celery Task — Sample Generation from Trained DDPM
=====================================================
Loads trained model weights and runs the reverse diffusion process
to generate brand-new synthetic sequences from pure noise.
"""

import torch
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from app.tasks.celery_app import celery_app
from app.tasks.db import SyncSession
from app.models.experiment import Experiment
from app.services.unet import UNet1D
from app.services.ddpm_math import make_beta_schedule, precompute_schedule_values
from app.services.training import sample_from_model
from app.services.financial_data import compute_stylized_facts


@celery_app.task(bind=True, name="generate_samples_task")
def generate_samples_task(self, experiment_id: str, n_samples: int = 4):
    """
    Load the trained model for this experiment and generate n_samples
    new synthetic sequences via the reverse diffusion process.
    Stores results in experiment.generated_samples under a new key.
    """
    session = SyncSession()

    try:
        experiment = session.execute(
            select(Experiment).where(Experiment.id == experiment_id)
        ).scalar_one_or_none()

        if not experiment or not experiment.model_weights_path:
            return {"error": "No trained model found for this experiment"}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rebuild model architecture and load trained weights
        model = UNet1D(d_model=experiment.d_model).to(device)
        state_dict = torch.load(experiment.model_weights_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        # Rebuild the noise schedule (needed for reverse process math)
        betas = make_beta_schedule(
            T=experiment.T, beta_start=experiment.beta_start, beta_end=experiment.beta_end
        )
        schedule = precompute_schedule_values(betas)
        schedule = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in schedule.items()}

        # Run reverse diffusion — this is the actual generation process
        generated = sample_from_model(
            model=model,
            schedule=schedule,
            sequence_length=experiment.sequence_length,
            n_samples=n_samples,
            device=device,
        )

        # Store results
        # Compute stylized facts on the generated samples (only meaningful
        # for financial-data experiments, but harmless to compute regardless)
        generated_np = generated.cpu().numpy()
        stylized_facts_generated = compute_stylized_facts(generated_np)

        # Store results
        current_samples = dict(experiment.generated_samples) if experiment.generated_samples else {}
        current_samples["model_generated"] = {
            "sequences": [
                [round(v, 4) for v in seq.tolist()] for seq in generated.cpu()
            ]
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