import uuid
from datetime import datetime, timezone
from sqlalchemy import String, Float, Integer, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum
from app.db.session import Base


class ExperimentStatus(str, enum.Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=True)

    # DDPM hyperparameters
    T: Mapped[int] = mapped_column(Integer, default=1000)
    beta_start: Mapped[float] = mapped_column(Float, default=1e-4)
    beta_end: Mapped[float] = mapped_column(Float, default=0.02)
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    batch_size: Mapped[int] = mapped_column(Integer, default=64)
    learning_rate: Mapped[float] = mapped_column(Float, default=2e-4)
    sequence_length: Mapped[int] = mapped_column(Integer, default=128)
    d_model: Mapped[int] = mapped_column(Integer, default=64)

    # Status and results
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus), default=ExperimentStatus.PENDING
    )
    current_epoch: Mapped[int] = mapped_column(Integer, default=0)
    train_losses: Mapped[list] = mapped_column(JSON, default=list)
    final_loss: Mapped[float] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # SNR, stylized facts etc.
    generated_samples: Mapped[dict] = mapped_column(JSON, default=dict)  # plot data

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)

    # relationships
    user: Mapped["User"] = relationship(back_populates="experiments")