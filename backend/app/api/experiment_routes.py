from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio

from fastapi import APIRouter, Depends, status, HTTPException
from app.db.session import get_db
from app.api.deps import get_current_user
from app.models.user import User
from app.schemas.experiment import ExperimentCreate, ExperimentOut
from app.services.experiment_service import (
    create_and_run_phase1_experiment,
    get_experiment,
    list_experiments,
    delete_experiment,
)
from app.tasks.training_tasks import train_ddpm_task
from app.tasks.generation_tasks import generate_samples_task
from app.core.config import settings
from app.services.task_runner import execute_training, execute_generation

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/", response_model=ExperimentOut, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    data: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = await create_and_run_phase1_experiment(db, current_user.id, data)
    return experiment


@router.get("/", response_model=list[ExperimentOut])
async def get_all_experiments(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await list_experiments(db, current_user.id)


@router.get("/{experiment_id}", response_model=ExperimentOut)
async def get_one_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return await get_experiment(db, current_user.id, experiment_id)


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_experiment(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await delete_experiment(db, current_user.id, experiment_id)

@router.post("/{experiment_id}/train", status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    experiment_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = await get_experiment(db, current_user.id, experiment_id)

    if settings.USE_CELERY:
        task = train_ddpm_task.delay(experiment_id)
        return {"message": "Training started", "task_id": task.id}
    else:
        if experiment.epochs > 5 or experiment.sequence_length > 32 or experiment.d_model > 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "This demo environment runs on a free-tier server with limited memory. "
                    "For live training here, please use epochs ≤ 5, sequence_length ≤ 32, "
                    "and d_model ≤ 8. Larger runs are demonstrated in the project README/video."
                ),
            )
        # Run in a background thread so it doesn't block the single-worker
        # event loop — the free-tier deployment has no Celery, but we still
        # need the server to stay responsive to other requests (like status
        # polling) while training runs.
        asyncio.create_task(asyncio.to_thread(execute_training, experiment_id))
        return {"message": "Training started in background"}
    

@router.post("/{experiment_id}/generate", status_code=status.HTTP_202_ACCEPTED)
async def start_generation(
    experiment_id: str,
    n_samples: int = 4,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    experiment = await get_experiment(db, current_user.id, experiment_id)
    if not experiment.model_weights_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model must be trained before generating samples",
        )

    if settings.USE_CELERY:
        task = generate_samples_task.delay(experiment_id, n_samples)
        return {"message": "Generation started", "task_id": task.id}
    else:
        result = execute_generation(experiment_id, n_samples)
        return {"message": "Generation completed", "result": result}