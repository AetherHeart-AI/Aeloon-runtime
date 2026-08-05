"""Configuration owned by the optional Aeloon Cloud integration."""

from pydantic import BaseModel, ConfigDict


class CloudConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    base_url: str = "https://api.aetherheart.com"
    proxy: str | None = None
    device_name: str = "Aeloon Core"
    allow_insecure_http: bool = False


__all__ = ["CloudConfig"]
