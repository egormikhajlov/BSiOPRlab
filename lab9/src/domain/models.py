from dataclasses import dataclass
from typing import List


@dataclass
class Pattern:
    """Модель образа (буквы)."""
    name: str
    matrix: List[List[int]]   # 10x10, значения 0 или 1


@dataclass
class NoiseTest:
    """Результат теста на одном уровне зашумления."""
    noise_percent: int
    pattern_name: str
    trials: int
    success: int
    accuracy: float