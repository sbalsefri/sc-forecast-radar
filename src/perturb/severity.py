from dataclasses import dataclass


@dataclass(frozen=True)
class PerturbSpec:
    name: str
    severity: float
