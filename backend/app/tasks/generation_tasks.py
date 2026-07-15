"""
Celery Task — thin wrapper around the shared execute_generation() logic
in app.services.task_runner.
"""

from app.tasks.celery_app import celery_app
from app.services.task_runner import execute_generation


@celery_app.task(bind=True, name="generate_samples_task")
def generate_samples_task(self, experiment_id: str, n_samples: int = 4):
    return execute_generation(experiment_id, n_samples)