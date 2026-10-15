from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Criterion:
    """Модель критерия."""
    id: str
    name: str
    weight: float = 0.0
    is_maximize: bool = True  # True — чем больше, тем лучше; False — чем меньше


@dataclass
class Alternative:
    """Модель альтернативы (станка)."""
    id: str
    name: str
    scores: Dict[str, float]  # оценки по критериям (id критерия -> значение)