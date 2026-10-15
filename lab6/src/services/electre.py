from typing import Dict, List, Tuple

from src.domain.models import Alternative, Criterion


class ElectreDecisionMaker:
    """
    Математическое ядро метода ЭЛЕКТРА I.
    Реализует три этапа:
      1. Отбор Парето-оптимальных альтернатив.
      2. Нормирование критериев.
      3. Расчёт индексов согласия/несогласия и построение ядра.
    """

    def __init__(self, alternatives: List[Alternative], criteria: List[Criterion]):
        self.alternatives = alternatives
        self.criteria = criteria

    # ---------- ЭТАП 1: МНОЖЕСТВО ПАРЕТО ----------

    def get_pareto_optimal_set(self) -> List[Alternative]:
        """
        Возвращает список Парето-оптимальных альтернатив.
        Альтернатива A доминируется B, если:
          - B не хуже A по всем критериям,
          - B строго лучше A хотя бы по одному.
        """
        pareto: List[Alternative] = []

        for i, a in enumerate(self.alternatives):
            is_dominated = False
            for j, b in enumerate(self.alternatives):
                if i == j:
                    continue

                better_or_equal = True
                strictly_better = False

                for crit in self.criteria:
                    va = a.scores[crit.id]
                    vb = b.scores[crit.id]

                    # Приводим к виду "чем больше, тем лучше"
                    if not crit.is_maximize:
                        va = -va
                        vb = -vb

                    if vb < va:
                        better_or_equal = False
                        break
                    if vb > va:
                        strictly_better = True

                if better_or_equal and strictly_better:
                    is_dominated = True
                    break

            if not is_dominated:
                pareto.append(a)

        return pareto

    # ---------- ЭТАП 2: НОРМИРОВАНИЕ ----------

    def _normalize(self, alts: List[Alternative]) -> List[Dict[str, float]]:
        """
        Приводит оценки к безразмерному виду [0, 1].
        Для критериев "чем больше, тем лучше" — делим на максимум.
        Для критериев "чем меньше, тем лучше" — берём обратное значение,
        затем нормируем на максимум среди обратных.
        """
        # Считаем преобразованные значения
        converted: List[Dict[str, float]] = []
        for alt in alts:
            row: Dict[str, float] = {}
            for crit in self.criteria:
                v = alt.scores[crit.id]
                if crit.is_maximize:
                    row[crit.id] = v
                else:
                    # инвертируем: чем меньше, тем лучше
                    row[crit.id] = 1.0 / v if v != 0 else 0.0
            converted.append(row)

        # Нормируем каждый критерий на максимум
        max_vals: Dict[str, float] = {}
        for crit in self.criteria:
            max_vals[crit.id] = max(row[crit.id] for row in converted)

        normalized: List[Dict[str, float]] = []
        for row in converted:
            norm_row: Dict[str, float] = {}
            for crit in self.criteria:
                m = max_vals[crit.id]
                norm_row[crit.id] = row[crit.id] / m if m != 0 else 0.0
            normalized.append(norm_row)

        return normalized

    # ---------- ЭТАП 3: ЭЛЕКТРА I ----------

    def apply_electre(
        self,
        active_alts: List[Alternative],
        weights: Dict[str, float],
        c_threshold: float = 0.5,
        d_threshold: float = 0.5
    ) -> dict:
        """
        Реализация метода ЭЛЕКТРА I.
        Возвращает:
          - матрицу индексов согласия,
          - матрицу индексов несогласия,
          - ядро (лучшие альтернативы),
          - нормализованные оценки.
        """
        n = len(active_alts)
        normalized = self._normalize(active_alts)

        # Индексы согласия и несогласия
        c_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        d_matrix: List[List[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c_val = 0.0
                max_d = 0.0
                for crit in self.criteria:
                    w = weights[crit.id]
                    vi = normalized[i][crit.id]
                    vj = normalized[j][crit.id]
                    if vi >= vj:
                        # критерий подтверждает превосходство i над j
                        c_val += w
                    else:
                        # критерий противоречит — считаем разрыв
                        diff = vj - vi
                        if diff > max_d:
                            max_d = diff
                c_matrix[i][j] = round(c_val, 3)
                d_matrix[i][j] = round(max_d, 3)

        # Определение ядра
        # Альтернатива j исключается, если существует i такая, что:
        #   c[i][j] >= c* и d[i][j] <= d*
        dominated = [False] * n
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if c_matrix[i][j] >= c_threshold and d_matrix[i][j] <= d_threshold:
                    dominated[j] = True

        kernel = [active_alts[i].name for i in range(n) if not dominated[i]]
        if not kernel:
            # Если пороги слишком строгие — берём альтернативу с макс. суммой c
            best_i = max(range(n), key=lambda k: sum(c_matrix[k]))
            kernel = [active_alts[best_i].name]

        return {
            "alternatives": [a.name for a in active_alts],
            "normalized": [
                {crit.id: round(normalized[i][crit.id], 3) for crit in self.criteria}
                for i in range(n)
            ],
            "c_matrix": c_matrix,
            "d_matrix": d_matrix,
            "kernel": kernel,
        }