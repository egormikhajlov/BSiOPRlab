from dataclasses import dataclass
from typing import List

@dataclass
class SimulationConfig:
    """Параметры моделирования."""
    n: int = 10000              # количество партий
    seed: int = 0xACE1          # начальное состояние LFSR
    a_percent: float = 25.0     # доля бракованных партий, %
    cost_control: float = 2.0   # B, руб.
    cost_fix: float = 65.0      # C, руб.
    cost_replace: float = 110.0 # D, руб.
    step: int = 5               # шаг по проценту контроля

@dataclass
class PointResult:
    """Результат для одного процента контроля."""
    control_percent: int
    total_cost: float
    cost_per_party: float

@dataclass
class SimulationResult:
    """Общий результат моделирования."""
    config: SimulationConfig
    points: List[PointResult]
    best_percent: int
    best_cost: float