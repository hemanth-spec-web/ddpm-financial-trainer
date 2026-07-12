# DDPM Trainer — Full-Stack Diffusion Model Platform

A production-style full-stack application for building, training, and visualizing
Denoising Diffusion Probabilistic Models (DDPM) from scratch — implementing the
mathematics from Ho et al. 2020 with a real authentication system, database
persistence, and interactive visualizations.

Built as part of a portfolio project bridging Edge AI, quantitative research,
and full-stack engineering.

---

## What This Is

Most DDPM tutorials are a single Jupyter notebook. This project instead treats
diffusion model training as a real product: users register, create experiments
with custom hyperparameters, and see their results — noise schedules, forward
process visualizations, and (in later phases) actual training curves — persisted
per-account and rendered as interactive charts.

## Architecture

ddpm-app/
├── backend/                  FastAPI + PostgreSQL + SQLAlchemy (async)
│   ├── app/
│   │   ├── api/               Route handlers (auth, experiments)
│   │   ├── core/               Config, JWT security
│   │   ├── db/                  Async DB session, base
│   │   ├── models/              SQLAlchemy ORM models (User, Experiment, RefreshToken)
│   │   ├── schemas/             Pydantic request/response schemas
│   │   └── services/            Business logic (auth, DDPM math, experiment orchestration)
│   └── alembic/                Database migrations
├── frontend/                 React + Vite + Recharts
│   └── src/
│       ├── api/                  Axios clients with auto token refresh
│       ├── components/         Protected route wrapper
│       ├── pages/                Login, Register, Dashboard, ExperimentDetail
│       └── store/                Zustand auth state
└── docker-compose.yml        Postgres + backend + frontend orchestration

## Features

**Authentication**
- JWT access + refresh tokens with automatic silent refresh on expiry
- Bcrypt password hashing
- Protected API routes and frontend routes

**DDPM Mathematics (Phase 1 — complete)**
- Linear beta noise schedule (Ho et al. 2020, Section 2)
- Closed-form forward process: `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`
- 5 automated correctness tests (signal dominance at t=0, pure noise at t=T,
  monotonicity, bounds checking) run on every experiment
- Interactive Recharts visualization of signal/noise coefficient decay,
  beta schedule, and the forward process applied to a synthetic time series

**Experiment Tracking**
- Per-user experiment history stored in PostgreSQL
- Custom hyperparameters per run (T, β_start, β_end, sequence length, etc.)
- Full audit trail: status (pending → running → completed/failed), timestamps

**Coming in later phases**
- 1D U-Net denoiser architecture
- Celery + Redis background training jobs (training runs take minutes to hours,
  so this moves off the synchronous request path)
- Live training loss curves via polling/WebSocket
- Application to real financial time-series (S&P 500 returns via yfinance)
- Stylized-fact evaluation (fat tails, volatility clustering) for the quant angle

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, asyncpg |
| Database | PostgreSQL 16 |
| Auth | JWT (python-jose), bcrypt (passlib) |
| ML | PyTorch, NumPy |
| Frontend | React 18, Vite, Recharts, Zustand, React Router |
| Infra | Docker Compose |

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, Docker Desktop

```bash
# 1. Start Postgres
docker compose up postgres -d

# 2. Backend
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# 3. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to use the app, or `http://localhost:8000/docs`
for the interactive API documentation.

## Why This Project

Diffusion models power most modern generative AI (Stable Diffusion, Sora, and
related systems). Understanding the mathematics from first principles — rather
than calling a pre-built pipeline — demonstrates the kind of deep technical
grounding that research teams and quant firms look for. This project pairs that
mathematical rigor with the software engineering discipline (auth, databases,
API design, testing) expected in production ML systems.

## Author

Hemanth — B.Tech ECE, NIT Warangal