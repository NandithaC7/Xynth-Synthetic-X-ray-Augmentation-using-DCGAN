"""Application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root: Xynth/ (parent of backend/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for training, generation, and the API."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dataset_path: str = "COVID-19_Radiography_Dataset"
    output_path: str = "backend/outputs"
    target_class: str = "COVID"
    latent_dim: int = 100
    image_size: int = 64
    batch_size: int = 64
    epochs: int = 50
    classifier_epochs: int = 10
    num_generated: int = 500
    checkpoint_every: int = 5
    learning_rate: float = 0.0002
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # Class names matching the COVID-19 Radiography Database
    class_names: tuple[str, ...] = (
        "COVID",
        "Normal",
        "Viral Pneumonia",
        "Lung_Opacity",
    )

    @property
    def dataset_dir(self) -> Path:
        """Absolute path to the radiography dataset root."""
        path = Path(self.dataset_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def output_dir(self) -> Path:
        """Absolute path to the outputs directory."""
        path = Path(self.output_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def generated_dir(self) -> Path:
        return self.output_dir / "generated"

    @property
    def checkpoints_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def augmented_dir(self) -> Path:
        return PROJECT_ROOT / "augmented_dataset"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
