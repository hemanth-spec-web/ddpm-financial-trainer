from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from app.models.experiment import ExperimentStatus


class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    T: int = Field(1000, ge=100, le=2000)
    beta_start: float = Field(1e-4, gt=0, lt=0.1)
    beta_end: float = Field(0.02, gt=0, lt=0.5)
    epochs: int = Field(100, ge=1, le=1000)
    batch_size: int = Field(64, ge=8, le=256)
    learning_rate: float = Field(2e-4, gt=0, lt=0.1)
    sequence_length: int = Field(128, ge=32, le=512)
    d_model: int = Field(64, ge=4, le=256)
    data_source: str = Field("synthetic", pattern="^(synthetic|financial)$")
    ticker: str = Field("^GSPC", max_length=20)


class ExperimentOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    T: int
    beta_start: float
    beta_end: float
    epochs: int
    batch_size: int
    learning_rate: float
    sequence_length: int
    d_model: int
    data_source: str
    ticker: Optional[str]
    status: ExperimentStatus
    current_epoch: int
    train_losses: list
    final_loss: Optional[float]
    metrics: dict
    generated_samples: dict
    model_weights_path: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]

    model_config = {"from_attributes": True}