from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AlertContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    provider: str
    model: str
    category: str
    window_runs: list[int]
    threshold: float | None = None


class DetectorResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    detector: Literal["pass_rate", "length", "latency", "text", "naive", "cusum"]
    metric_unit: Literal["points", "percent", "tokens", "seconds", "score", "none"]
    statistic: float | None = None
    p_value: float | None = Field(default=None, ge=0, le=1)
    effect_size: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reason: str
    status: Literal[
        "ok", "alarm", "no_data", "not_enough_power", "no_threshold", "warming_up"
    ]
    context: AlertContext
