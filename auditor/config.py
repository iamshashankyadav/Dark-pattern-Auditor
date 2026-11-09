"""Runtime configuration, read from environment variables / a .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'auditor.db').as_posix()}"

    # Phase 2 backend: "setfit" or "zero_shot". If the SetFit model has not been
    # trained yet the pipeline falls back to zero-shot automatically.
    classifier_backend: str = "setfit"
    zero_shot_model: str = "facebook/bart-large-mnli"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    setfit_model_dir: Path = PROJECT_ROOT / "models" / "setfit"
    span_model_dir: Path = PROJECT_ROOT / "models" / "span_extractor"

    # Phase 2 routing: violations at or above this confidence are flagged,
    # anything less certain goes to the human evaluation queue.
    flag_threshold: float = 0.70
    # Violations scoring between review_threshold and flag_threshold are uncertain and
    # are sent to human reviewers; their decisions feed SetFit retraining.
    review_threshold: float = 0.50
    # Phase 1: cosine similarity to the risk-phrase bank that marks text as suspicious.
    semantic_threshold: float = 0.55

    device: int = -1  # -1 = auto (GPU if available)


@lru_cache
def get_settings() -> Settings:
    return Settings()
