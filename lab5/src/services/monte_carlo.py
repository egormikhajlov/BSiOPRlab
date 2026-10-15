from typing import List
from src.domain.models import SimulationConfig, PointResult, SimulationResult
from src.services.lfsr import LFSR


class MonteCarloSimulator:
    """Моделирование процесса контроля и подгонки резисторов."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def _simulate_for_percent(self, control_percent: int) -> PointResult:
        """
        Моделирует N партий при заданном проценте контроля.
        Возвращает суммарные затраты.
        """
        cfg = self.config
        p = control_percent / 100.0
        p_defect = cfg.a_percent / 100.0

        # Инициализируем LFSR заново для каждого процента,
        # чтобы результаты были воспроизводимы и независимы.
        lfsr = LFSR(seed=cfg.seed)

        total_cost = 0.0
        for _ in range(cfg.n):
            r1 = lfsr.next_float()
            r2 = lfsr.next_float()

            if r1 < p:
                # Партия отправлена на контроль
                total_cost += cfg.cost_control
                if r2 < p_defect:
                    # Обнаружен брак -> подгонка
                    total_cost += cfg.cost_fix
            else:
                # Партия без контроля
                if r2 < p_defect:
                    # Брак обнаружится при установке -> замена
                    total_cost += cfg.cost_replace

        return PointResult(
            control_percent=control_percent,
            total_cost=total_cost,
            cost_per_party=total_cost / cfg.n
        )

    def run(self) -> SimulationResult:
        """Прогоняет моделирование для всех процентов контроля."""
        points: List[PointResult] = []
        for percent in range(0, 101, self.config.step):
            points.append(self._simulate_for_percent(percent))

        best = min(points, key=lambda p: p.total_cost)

        return SimulationResult(
            config=self.config,
            points=points,
            best_percent=best.control_percent,
            best_cost=best.total_cost
        )