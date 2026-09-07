"""FastAPI application exposing dataset, training, generation, and results APIs."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.utils import ensure_dirs, load_json
from backend.data.dataset_loader import get_dataset_stats

settings = get_settings()
ensure_dirs(
    settings.generated_dir,
    settings.checkpoints_dir,
    settings.metrics_dir,
    settings.plots_dir,
)

app = FastAPI(
    title="Xynth API",
    description="Synthetic X-ray augmentation using DCGAN",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images and plots for the frontend gallery
app.mount(
    "/static/generated",
    StaticFiles(directory=str(settings.generated_dir)),
    name="generated",
)
app.mount(
    "/static/plots",
    StaticFiles(directory=str(settings.plots_dir)),
    name="plots",
)


# ---------------------------------------------------------------------------
# In-memory job tracker (simple background workers)
# ---------------------------------------------------------------------------

JobName = Literal["dcgan", "generate", "classifier"]

_jobs_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {
    "dcgan": {"status": "idle", "message": "", "updated_at": None, "result": None},
    "generate": {"status": "idle", "message": "", "updated_at": None, "result": None},
    "classifier": {"status": "idle", "message": "", "updated_at": None, "result": None},
}


def _set_job(name: JobName, **kwargs: Any) -> None:
    with _jobs_lock:
        _jobs[name].update(kwargs)
        _jobs[name]["updated_at"] = datetime.now(timezone.utc).isoformat()


def _get_job(name: JobName) -> dict[str, Any]:
    with _jobs_lock:
        return dict(_jobs[name])


class TrainDCGANRequest(BaseModel):
    """Optional overrides for DCGAN training."""

    epochs: int | None = Field(default=None, ge=1, le=500)
    batch_size: int | None = Field(default=None, ge=1, le=256)
    resume: bool = False


class GenerateRequest(BaseModel):
    """Optional overrides for synthetic image generation."""

    num_images: int | None = Field(default=None, ge=1, le=10000)
    build_augmented: bool = True


class TrainClassifierRequest(BaseModel):
    """Optional overrides for CNN training."""

    epochs: int | None = Field(default=None, ge=1, le=200)
    max_per_class: int | None = Field(default=None, ge=10, le=20000)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok", "project": "Xynth"}


@app.get("/dataset/stats")
def dataset_stats() -> dict[str, Any]:
    """Return per-class counts for the configured dataset."""
    return get_dataset_stats(settings)


@app.get("/jobs/{name}")
def job_status(name: JobName) -> dict[str, Any]:
    """Return status of a background training / generation job."""
    return {"name": name, **_get_job(name)}


def _run_dcgan(epochs: int | None, batch_size: int | None, resume: bool) -> None:
    try:
        _set_job("dcgan", status="running", message="Training DCGAN…", result=None)
        from backend.training.train_dcgan import train_dcgan

        result = train_dcgan(epochs=epochs, batch_size=batch_size, resume=resume)
        _set_job("dcgan", status="completed", message="DCGAN training finished", result=result)
    except Exception as exc:  # noqa: BLE001
        _set_job("dcgan", status="failed", message=str(exc), result=None)


@app.post("/train/dcgan")
def train_dcgan_endpoint(
    body: TrainDCGANRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """
    Start DCGAN training in a background task.

    Returns immediately with job status. Poll GET /jobs/dcgan for progress.
    """
    job = _get_job("dcgan")
    if job["status"] == "running":
        return {"status": "already_running", "job": job}

    background_tasks.add_task(_run_dcgan, body.epochs, body.batch_size, body.resume)
    _set_job("dcgan", status="queued", message="DCGAN job queued", result=None)
    return {
        "status": "queued",
        "message": "DCGAN training started in background",
        "epochs": body.epochs or settings.epochs,
        "job": _get_job("dcgan"),
    }


def _run_generate(num_images: int | None, build_augmented: bool) -> None:
    try:
        _set_job("generate", status="running", message="Generating images…", result=None)
        from backend.training.generate_images import build_augmented_dataset, generate_images

        result = generate_images(num_images=num_images)
        if build_augmented:
            aug = build_augmented_dataset(num_synthetic=num_images)
            result["augmented"] = aug
        _set_job("generate", status="completed", message="Generation finished", result=result)
    except Exception as exc:  # noqa: BLE001
        _set_job("generate", status="failed", message=str(exc), result=None)


@app.post("/generate")
def generate_endpoint(
    body: GenerateRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Generate synthetic images (and optionally build the augmented dataset)."""
    job = _get_job("generate")
    if job["status"] == "running":
        return {"status": "already_running", "job": job}

    background_tasks.add_task(_run_generate, body.num_images, body.build_augmented)
    _set_job("generate", status="queued", message="Generation job queued", result=None)
    return {
        "status": "queued",
        "message": "Generation started in background",
        "num_images": body.num_images or settings.num_generated,
        "job": _get_job("generate"),
    }


def _run_classifier(epochs: int | None, max_per_class: int | None) -> None:
    try:
        _set_job("classifier", status="running", message="Training classifiers…", result=None)
        from backend.training.train_classifier import train_classifiers

        result = train_classifiers(epochs=epochs, max_per_class=max_per_class)
        _set_job("classifier", status="completed", message="Classifier training finished", result=result)
    except Exception as exc:  # noqa: BLE001
        _set_job("classifier", status="failed", message=str(exc), result=None)


@app.post("/train/classifier")
def train_classifier_endpoint(
    body: TrainClassifierRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Train baseline and augmented CNN classifiers in the background."""
    job = _get_job("classifier")
    if job["status"] == "running":
        return {"status": "already_running", "job": job}

    background_tasks.add_task(_run_classifier, body.epochs, body.max_per_class)
    _set_job("classifier", status="queued", message="Classifier job queued", result=None)
    return {
        "status": "queued",
        "message": "Classifier training started in background",
        "epochs": body.epochs or settings.classifier_epochs,
        "job": _get_job("classifier"),
    }


@app.get("/results")
def results() -> dict[str, Any]:
    """Aggregate metrics, plots, checkpoints, and sample galleries for the UI."""
    metrics_dir = settings.metrics_dir
    plots_dir = settings.plots_dir
    generated_dir = settings.generated_dir
    checkpoints_dir = settings.checkpoints_dir

    comparison = load_json(metrics_dir / "comparison.json")
    baseline = load_json(metrics_dir / "classifier_baseline.json")
    augmented = load_json(metrics_dir / "classifier_augmented.json")
    dcgan = load_json(metrics_dir / "dcgan_training.json")
    losses = load_json(metrics_dir / "dcgan_losses.json")
    generation = load_json(metrics_dir / "generation.json")
    aug_meta = load_json(metrics_dir / "augmented_dataset.json")

    # Downsample losses for charting if very long
    loss_payload = None
    if losses:
        g = losses.get("g_losses", [])
        d = losses.get("d_losses", [])
        step = max(1, len(g) // 500)
        loss_payload = {
            "g_losses": g[::step],
            "d_losses": d[::step],
            "total_steps": len(g),
        }

    latest_ckpt = checkpoints_dir / "dcgan_latest.pt"
    checkpoint_info = None
    if latest_ckpt.exists():
        import torch

        state = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        checkpoint_info = {
            "path": str(latest_ckpt),
            "epoch": state.get("epoch"),
            "target_class": state.get("target_class"),
            "latent_dim": state.get("latent_dim"),
            "modified": datetime.fromtimestamp(
                latest_ckpt.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
        }

    sample_files = sorted((generated_dir / "samples").glob("*.png")) if (generated_dir / "samples").exists() else []
    gallery = [f"/static/generated/samples/{p.name}" for p in sample_files[:48]]

    grids = []
    for name in ("preview_collage.png", "preview_4x4.png"):
        if (generated_dir / name).exists():
            grids.append(f"/static/generated/{name}")
    for p in sorted(generated_dir.glob("epoch_*_grid.png"))[-6:]:
        grids.append(f"/static/generated/{p.name}")

    plot_urls = {}
    for name in ("dcgan_loss.png", "confusion_baseline.png", "confusion_augmented.png"):
        if (plots_dir / name).exists():
            plot_urls[name.replace(".png", "")] = f"/static/plots/{name}"

    return {
        "dataset": get_dataset_stats(settings),
        "comparison": comparison,
        "baseline": baseline,
        "augmented": augmented,
        "dcgan": dcgan,
        "losses": loss_payload,
        "generation": generation,
        "augmented_dataset": aug_meta,
        "checkpoint": checkpoint_info,
        "gallery": gallery,
        "grids": grids,
        "plots": plot_urls,
        "jobs": {k: _get_job(k) for k in ("dcgan", "generate", "classifier")},  # type: ignore[arg-type]
        "config": {
            "target_class": settings.target_class,
            "image_size": settings.image_size,
            "latent_dim": settings.latent_dim,
            "batch_size": settings.batch_size,
            "epochs": settings.epochs,
            "classifier_epochs": settings.classifier_epochs,
            "num_generated": settings.num_generated,
        },
    }


@app.get("/files/plot/{name}")
def get_plot(name: str) -> FileResponse:
    """Download a plot PNG by filename."""
    safe = Path(name).name
    path = settings.plots_dir / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(path)


def create_app() -> FastAPI:
    """Factory used by uvicorn / tests."""
    return app
