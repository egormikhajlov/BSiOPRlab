from dataclasses import dataclass
from typing import List


@dataclass
class DataObject:
    """Модель объекта для кластеризации (города)."""
    id: str
    features: List[float]   # 3 признака: население, зарплата, площадь
    name: str = ""          # название города