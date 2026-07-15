from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from app.models.experiment import Experiment, ExperimentStatus
from app.schemas.experiment import ExperimentCreate
from app.services.ddpm_math import (
    make_beta_schedule,
    precompute_schedule_values,
    generate_schedule_summary,
    generate_forward_process_demo,
    run_unit_tests,
)


async def create_and_run_phase1_experiment(
    db: AsyncSession, user_id: str, data: ExperimentCreate
) -> Experiment:
    """
    Create an experiment record and immediately run Phase 1
    (noise schedule + forward process) synchronously.
    """
    experiment = Experiment(
        user_id=user_id,
        name=data.name,
        description=data.description,
        T=data.T,
        beta_start=data.beta_start,
        beta_end=data.beta_end,
        epochs=data.epochs,
        batch_size=data.batch_size,
        learning_rate=data.learning_rate,
        sequence_length=data.sequence_length,
        d_model=data.d_model,
        data_source=data.data_source,
        ticker=data.ticker,
        status=ExperimentStatus.RUNNING,
    )                       
    db.add(experiment)
    await db.flush()

    try:
        betas = make_beta_schedule(T=data.T, beta_start=data.beta_start, beta_end=data.beta_end)
        schedule = precompute_schedule_values(betas)

        test_results = run_unit_tests(schedule)
        schedule_summary = generate_schedule_summary(schedule)
        forward_demo = generate_forward_process_demo(schedule, sequence_length=data.sequence_length)

        experiment.metrics = {
            "unit_tests": test_results,
            "schedule_summary": schedule_summary,
        }
        experiment.generated_samples = forward_demo
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = datetime.now(timezone.utc)

    except Exception as e:
        experiment.status = ExperimentStatus.FAILED
        experiment.metrics = {"error": str(e)}

    await db.flush()
    await db.refresh(experiment)
    return experiment


async def get_experiment(db: AsyncSession, user_id: str, experiment_id: str) -> Experiment:
    result = await db.execute(
        select(Experiment).where(
            Experiment.id == experiment_id,
            Experiment.user_id == user_id,
        )
    )
    experiment = result.scalar_one_or_none()
    if not experiment:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    return experiment


async def list_experiments(db: AsyncSession, user_id: str) -> list[Experiment]:
    result = await db.execute(
        select(Experiment)
        .where(Experiment.user_id == user_id)
        .order_by(Experiment.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_experiment(db: AsyncSession, user_id: str, experiment_id: str) -> None:
    experiment = await get_experiment(db, user_id, experiment_id)
    await db.delete(experiment)
    await db.flush()