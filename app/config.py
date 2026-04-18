"""Application configuration from environment variables."""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from `.env` and the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    scholera_data_dir: Path = Field(default=Path("./data"), alias="SCHOLERA_DATA_DIR")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    google_api_key: str | None = Field(default=None, alias="GOOGLE_API_KEY")
    llm_model: str = Field(default="gemini-2.5-flash", alias="LLM_MODEL")
    hybrid_weight_dense: float = Field(default=0.65, alias="HYBRID_WEIGHT_DENSE")
    hybrid_weight_bm25: float = Field(default=0.35, alias="HYBRID_WEIGHT_BM25")

    @field_validator("scholera_data_dir", mode="before")
    @classmethod
    def _coerce_path(cls, v: str | Path) -> Path:
        return Path(v).expanduser().resolve()

    @property
    def courses_dir(self) -> Path:
        return self.scholera_data_dir / "courses"

    @property
    def hybrid_weights(self) -> tuple[float, float]:
        d, b = self.hybrid_weight_dense, self.hybrid_weight_bm25
        s = d + b
        if s <= 0:
            return 0.65, 0.35
        return d / s, b / s


def get_settings() -> Settings:
    """Lazily construct settings (useful for tests with env overrides)."""
    return Settings()
