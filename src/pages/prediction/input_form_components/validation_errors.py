from dataclasses import dataclass
from typing import Literal


@dataclass
class ValidationError:
    field: str
    message: str
    severity: Literal["error", "warning"] = "error"

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "message": self.message,
            "severity": self.severity,
        }

