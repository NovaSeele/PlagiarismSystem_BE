from pydantic import BaseModel
from typing import Optional


class LogEntry(BaseModel):
    """Model for log entries sent via WebSocket"""

    module: str
    message: str
    level: str = "info"
    timestamp: float
