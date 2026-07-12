from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

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
    task = train_ddpm_task.delay(experiment_id)
    return {"message": "Training started", "task_id": task.id}

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
    task = generate_samples_task.delay(experiment_id, n_samples)
    return {"message": "Generation started", "task_id": task.id}