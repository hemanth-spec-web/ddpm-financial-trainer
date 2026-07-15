"""
Celery Task — thin wrapper around the shared execute_training() logic
in app.services.task_runner. Keeping this file thin means the same
logic works identically whether invoked via Celery (local dev) or
directly as a background task (free-tier deployment).
"""

from app.tasks.celery_app import celery_app
from app.services.task_runner import execute_training


@celery_app.task(bind=True, name="train_ddpm_task")
def train_ddpm_task(self, experiment_id: str):
    return execute_training(experiment_id)