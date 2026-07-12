from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app import models  # noqa: F401
from app.api.auth_routes import router as auth_router
from app.api.experiment_routes import router as experiment_router

app = FastAPI(title="DDPM Trainer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(experiment_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "DDPM Trainer API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}