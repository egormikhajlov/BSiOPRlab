from dataclasses import dataclass
from typing import List


@dataclass
class DataObject:
    """Модель распознаваемого объекта (вина)."""
    id: str
    features: List[float]   # 5 признаков: крепость, сахар, кислотность, танинность, температура
    class_label: int        # 1 — красное, 2 — белое
    name: str = ""          # название вина