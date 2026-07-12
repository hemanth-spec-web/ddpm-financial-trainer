# DDPM Trainer — Full-Stack Diffusion Model Platform for Financial Time-Series

A production-shaped full-stack application implementing Denoising Diffusion
Probabilistic Models (DDPM) from scratch — from the raw mathematics (Ho et al.,
2020) through a trained neural network to a rigorous quantitative evaluation
against real S&P 500 market data.

Built as a portfolio project bridging Edge AI/ML engineering, quantitative
research, and full-stack software engineering.

---

## What This Is

Most DDPM projects stop at "here's a notebook that generates MNIST digits."
This one goes further in two directions at once:

1. **Full-stack engineering**: real authentication, PostgreSQL persistence,
   Celery background workers for long-running training jobs, and live
   progress polling — the same architectural shape as production ML systems.
2. **Quantitative rigor**: the model is trained on real S&P 500 returns
   (via yfinance) and evaluated against the "stylized facts" that any
   credible market simulator must reproduce (Cont, 2001) — fat tails,
   volatility clustering, and near-zero return autocorrelation.

## Results

Trained the same architecture for two epoch budgets to demonstrate the
model actually learning market statistics, not just memorizing noise:

| Stylized Fact | Real S&P 500 | Generated (5 epochs) | Generated (100 epochs) |
|---|---|---|---|
| Kurtosis (fat tails) | 2.10 | 15.52 (overshoot) | **2.51** (close match) |
| Skewness | -0.087 | +0.978 (wrong sign) | **-0.273** (correct sign) |
| Return autocorrelation (lag 1) | -0.008 | -0.089 | **0.005** (excellent) |
| Volatility clustering (avg) | 0.049 | 0.039 | tracked live in-app |

At 5 epochs, the model undershoots on tail calibration — a textbook sign
of undertraining. At 100 epochs, kurtosis and skewness converge close to
the real distribution's values, while return autocorrelation stays
correctly near zero throughout (the model never learns spurious
predictability, which would be the actual failure mode to worry about).

## Architecture

ddpm-app/
├── backend/                     FastAPI + PostgreSQL + Celery + Redis
│   ├── app/
│   │   ├── api/                   Route handlers (auth, experiments)
│   │   ├── core/                   Config, JWT security
│   │   ├── db/                       Async DB session, base
│   │   ├── models/                   SQLAlchemy ORM (User, Experiment, RefreshToken)
│   │   ├── schemas/                 Pydantic request/response schemas
│   │   ├── services/                 DDPM math, U-Net, training loop, financial data
│   │   └── tasks/                     Celery background jobs (train, generate)
│   ├── model_weights/             Saved .pt files per experiment
│   └── alembic/                    Database migrations
├── frontend/                    React + Vite + Recharts
│   └── src/
│       ├── api/                       Axios clients with auto token refresh
│       ├── pages/                     Login, Register, Dashboard, ExperimentDetail
│       └── store/                     Zustand auth state
└── docker-compose.yml            Postgres + Redis + backend + frontend

## Features

**Authentication**
JWT access + refresh tokens with automatic silent refresh, bcrypt password
hashing, protected API and frontend routes.

**DDPM Mathematics — implemented from scratch, not via a library**
- Linear beta noise schedule and closed-form forward process:
  `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`
- 1D U-Net denoiser (2.3M parameters): residual blocks with GroupNorm,
  sinusoidal timestep embeddings injected at every block, self-attention
  at the bottleneck, skip connections between encoder/decoder
- Full reverse diffusion sampling (Algorithm 2, Ho et al. 2020) — generates
  new sequences from pure Gaussian noise
- 5 automated correctness tests run on every experiment (signal dominance
  at t=0, pure noise at t=T, monotonicity, bounds checking)

**Real Financial Data Integration**
- Live S&P 500 (or any ticker) daily returns via yfinance
- Configurable per-experiment: synthetic sine-wave data or real market data
- Stylized facts evaluator: kurtosis, skewness, return autocorrelation,
  and volatility clustering — computed on both real training data and
  model-generated samples, displayed side-by-side

**Background Training Infrastructure**
- Celery + Redis run training and generation as background jobs so the
  API never blocks — training runs of any length (tested up to 100 epochs,
  ~15 minutes) don't tie up a request thread
- Live progress: current epoch and loss history written to PostgreSQL
  after every epoch, polled by the frontend every 2 seconds
- Trained model weights persisted to disk, reloadable for generation
  without retraining

**Experiment Tracking**
Per-user experiment history in PostgreSQL — every hyperparameter,
every loss value, every generated sample, and every evaluation metric
tied to a specific experiment and specific account.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, SQLAlchemy 2.0 (async), Alembic, asyncpg |
| Background Jobs | Celery, Redis |
| Database | PostgreSQL 16 |
| Auth | JWT (python-jose), bcrypt (passlib) |
| ML | PyTorch (custom U-Net, no pre-built diffusion library) |
| Financial Data | yfinance |
| Frontend | React 18, Vite, Recharts, Zustand, React Router |
| Infra | Docker Compose |

## Running Locally

**Prerequisites:** Python 3.11+, Node.js 18+, Docker Desktop

```bash
# 1. Start Postgres and Redis
docker compose up postgres redis -d

# 2. Backend API
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload

# 3. Celery worker (separate terminal)
cd backend
venv\Scripts\activate
celery -A app.tasks.celery_app worker --loglevel=info --pool=solo   # --pool=solo required on Windows

# 4. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to use the app, or `http://localhost:8000/docs`
for interactive API documentation.

## Why This Project

Diffusion models power most modern generative AI. Understanding the
mathematics well enough to implement it from a paper — rather than calling
a pre-built pipeline — demonstrates the technical depth research teams look
for. Pairing that with rigorous evaluation against real market data (not
just "does the loss go down") demonstrates the kind of statistical
discipline quantitative research teams specifically screen for. The full
authentication/database/background-job layer demonstrates the production
engineering skills expected of any ML engineering role.

## Author

Hemanth — B.Tech ECE, NIT Warangal