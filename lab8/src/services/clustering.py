import math
from typing import Dict, List, Tuple

from src.domain.models import DataObject


class ClusteringSystem:
    """
    Математическое ядро для кластерного анализа.
    Реализует два алгоритма:
      1. Метод К средних (K-means).
      2. Метод максимина (автоматическое определение числа кластеров).
    """

    def __init__(self, data: List[DataObject]):
        self.data = data
        self.feature_count = len(data[0].features)

    def _distance(self, f1: List[float], f2: List[float]) -> float:
        """Евклидово расстояние между двумя точками."""
        return math.sqrt(sum((f1[i] - f2[i]) ** 2 for i in range(self.feature_count)))

    # ---------- МЕТОД К СРЕДНИХ ----------

    def kmeans(self, k: int, max_iterations: int = 100) -> Tuple[Dict[int, List[DataObject]], List[List[float]], List[dict]]:
        """
        Метод К средних.
        Возвращает:
          - словарь кластеров (id кластера -> список объектов),
          - список прототипов (центров) на финальной итерации,
          - историю итераций (для отображения).
        """
        if not self.data or k <= 0 or k > len(self.data):
            return {}, [], []

        # Шаг 1: начальные прототипы = первые K объектов
        prototypes = [list(obj.features) for obj in self.data[:k]]
        history: List[dict] = []

        for iteration in range(1, max_iterations + 1):
            # Шаг 2: отнесение каждого объекта к ближайшему прототипу
            new_clusters: Dict[int, List[DataObject]] = {i: [] for i in range(k)}
            for obj in self.data:
                min_dist = float('inf')
                nearest = -1
                for i, p in enumerate(prototypes):
                    d = self._distance(obj.features, p)
                    if d < min_dist:
                        min_dist = d
                        nearest = i
                new_clusters[nearest].append(obj)

            # Шаг 3: пересчёт прототипов (среднее арифметическое)
            new_prototypes: List[List[float]] = []
            for i in range(k):
                cluster_objects = new_clusters[i]
                if not cluster_objects:
                    new_prototypes.append(prototypes[i])
                    continue
                avg = [0.0] * self.feature_count
                for obj in cluster_objects:
                    for j in range(self.feature_count):
                        avg[j] += obj.features[j]
                avg = [v / len(cluster_objects) for v in avg]
                new_prototypes.append(avg)

            # Записываем в историю
            history.append({
                "iteration": iteration,
                "prototypes": [[round(v, 2) for v in p] for p in new_prototypes],
                "cluster_sizes": [len(new_clusters[i]) for i in range(k)],
            })

            # Шаг 4: проверка сходимости
            converged = True
            for i in range(k):
                for j in range(self.feature_count):
                    if abs(prototypes[i][j] - new_prototypes[i][j]) > 1e-6:
                        converged = False
                        break
                if not converged:
                    break

            prototypes = new_prototypes

            if converged:
                return new_clusters, prototypes, history

        return new_clusters, prototypes, history

    # ---------- МЕТОД МАКСИМИНА ----------

    def maximin(self) -> Tuple[Dict[int, List[DataObject]], List[List[float]], List[dict]]:
        """
        Метод максимина. Число кластеров определяется автоматически.
        Возвращает словарь кластеров, список прототипов и историю шагов.
        """
        if not self.data:
            return {}, [], []

        history: List[dict] = []

        # Шаг 1: первый прототип — первый объект
        prototypes = [list(self.data[0].features)]
        history.append({
            "step": 1,
            "action": "Первый прототип — " + self.data[0].name,
            "prototypes": [prototypes[0][:]],
            "threshold": None,
        })

        # Шаг 2: второй прототип — самый удалённый от первого
        max_d = -1
        second_idx = 0
        for i, obj in enumerate(self.data):
            d = self._distance(prototypes[0], obj.features)
            if d > max_d:
                max_d = d
                second_idx = i

        prototypes.append(list(self.data[second_idx].features))
        history.append({
            "step": 2,
            "action": "Второй прототип — " + self.data[second_idx].name + f" (расстояние {max_d:.2f})",
            "prototypes": [p[:] for p in prototypes],
            "threshold": None,
        })

        # Шаг 3: пороговое расстояние
        threshold = self._distance(prototypes[0], prototypes[1]) / 2.0

        # Итерации
        max_steps = 50
        for step in range(max_steps):
            # Отнесение объектов к ближайшим прототипам
            clusters: Dict[int, List[DataObject]] = {i: [] for i in range(len(prototypes))}
            for obj in self.data:
                min_d = float('inf')
                nearest = -1
                for i, p in enumerate(prototypes):
                    d = self._distance(obj.features, p)
                    if d < min_d:
                        min_d = d
                        nearest = i
                clusters[nearest].append(obj)

            # Ищем самый удалённый объект в каждом кластере
            new_prototypes: List[List[float]] = []
            max_distances: List[float] = []
            for i, p in enumerate(prototypes):
                max_d = -1
                furthest = None
                for obj in clusters[i]:
                    d = self._distance(p, obj.features)
                    if d > max_d:
                        max_d = d
                        furthest = obj
                max_distances.append(max_d if furthest else 0)
                if furthest and max_d >= threshold:
                    new_prototypes.append((list(furthest.features), furthest.name))

            if not new_prototypes:
                # Алгоритм завершён
                history.append({
                    "step": len(history) + 1,
                    "action": f"Новых кластеров нет. Порог T = {threshold:.2f}",
                    "prototypes": [p[:] for p in prototypes],
                    "threshold": round(threshold, 2),
                })
                return clusters, prototypes, history

            # Добавляем новые прототипы
            for (proto, name) in new_prototypes:
                prototypes.append(proto)

            # Пересчитываем порог
            sum_d = 0.0
            count = 0
            for i in range(len(prototypes) - 1):
                for j in range(i + 1, len(prototypes)):
                    sum_d += self._distance(prototypes[i], prototypes[j])
                    count += 1
            if count > 0:
                threshold = (sum_d / count) / 2.0

            history.append({
                "step": len(history) + 1,
                "action": f"Добавлено {len(new_prototypes)} новых прототипов. Порог T = {threshold:.2f}",
                "prototypes": [p[:] for p in prototypes],
                "threshold": round(threshold, 2),
            })

        # Если вышли за лимит итераций
        clusters: Dict[int, List[DataObject]] = {i: [] for i in range(len(prototypes))}
        for obj in self.data:
            min_d = float('inf')
            nearest = -1
            for i, p in enumerate(prototypes):
                d = self._distance(obj.features, p)
                if d < min_d:
                    min_d = d
                    nearest = i
            clusters[nearest].append(obj)

        return clusters, prototypes, history