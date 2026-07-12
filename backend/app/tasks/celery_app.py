from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "ddpm_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,
)

celery_app.autodiscover_tasks(["app.tasks"])

# Explicit import ensures the task is registered even though the
# module isn't named tasks.py (autodiscover only looks for that name)
import app.tasks.training_tasks  # noqa: E402,F401
import app.tasks.generation_tasks  # noqa: E402,F401