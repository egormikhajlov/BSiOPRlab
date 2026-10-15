import math
from typing import List, Dict

from src.domain.models import DataObject


class RecognitionSystem:
    """
    Математическое ядро системы распознавания.
    Реализует два алгоритма:
      1. Метод минимального расстояния (через прототипы классов).
      2. Алгоритм восприятия (линейная разделяющая функция).
    """

    def __init__(self, training_data: List[DataObject]):
        self.training_data = training_data
        self.feature_count = len(training_data[0].features)

        # 1. Максимумы для нормирования
        self.max_values = self._find_max_values()

        # 2. Нормирование обучающей выборки
        self.norm_data = self._normalize_dataset(self.training_data)

        # 3. Прототипы классов для метода минимального расстояния
        self.prototypes = self._calculate_prototypes()

        # 4. Обучение перцептрона
        self.weights: List[float] = [0.0] * self.feature_count
        self.w0: float = 0.0
        self.history: List[int] = []   # количество ошибок на каждой итерации (для графика)
        self.iterations_done: int = 0
        self._train_perceptron()

    # ---------- НОРМИРОВАНИЕ ----------

    def _find_max_values(self) -> List[float]:
        max_vals = [0.0] * self.feature_count
        for obj in self.training_data:
            for i in range(self.feature_count):
                if obj.features[i] > max_vals[i]:
                    max_vals[i] = obj.features[i]
        return max_vals

    def _normalize_dataset(self, dataset: List[DataObject]) -> List[DataObject]:
        result = []
        for obj in dataset:
            norm_feats = [
                obj.features[i] / self.max_values[i] if self.max_values[i] != 0 else 0.0
                for i in range(self.feature_count)
            ]
            result.append(DataObject(obj.id, norm_feats, obj.class_label, obj.name))
        return result

    def normalize_single(self, features: List[float]) -> List[float]:
        return [
            features[i] / self.max_values[i] if self.max_values[i] != 0 else 0.0
            for i in range(self.feature_count)
        ]

    # ---------- МЕТОД МИНИМАЛЬНОГО РАССТОЯНИЯ ----------

    def _calculate_prototypes(self) -> Dict[int, List[float]]:
        """Средний вектор признаков для каждого класса."""
        prototypes = {1: [0.0] * self.feature_count, 2: [0.0] * self.feature_count}
        counts = {1: 0, 2: 0}

        for obj in self.norm_data:
            c = obj.class_label
            counts[c] += 1
            for i in range(self.feature_count):
                prototypes[c][i] += obj.features[i]

        for c in prototypes:
            if counts[c] > 0:
                prototypes[c] = [v / counts[c] for v in prototypes[c]]

        return prototypes

    def predict_min_distance(self, features_norm: List[float]) -> dict:
        """
        Решающая функция на основе минимального расстояния:
            D_i = 2 * Σ(X_j * P_ij) - Σ(P_ij^2)
        Класс — тот, у которого D_i максимальна.
        """
        d_values = {}
        for c, p in self.prototypes.items():
            sum_xp = sum(features_norm[i] * p[i] for i in range(self.feature_count))
            sum_pp = sum(p[i] * p[i] for i in range(self.feature_count))
            d_values[c] = 2 * sum_xp - sum_pp

        predicted = 1 if d_values[1] > d_values[2] else 2
        return {
            "predicted_class": predicted,
            "d1": round(d_values[1], 4),
            "d2": round(d_values[2], 4),
        }

    # ---------- АЛГОРИТМ ВОСПРИЯТИЯ (ПЕРЦЕПТРОН) ----------

    def _train_perceptron(self, max_iterations: int = 10000):
        """
        Обучение линейной разделяющей функции D12 = Σ(W_i * X_i) + W0.
        Правило: если объект 1-го класса распознан неверно (D12 <= 0),
                 W_i += X_i, W0 += 1.
                 Если 2-го класса неверно (D12 >= 0),
                 W_i -= X_i, W0 -= 1.
        Обучение завершается, когда все объекты распознаны правильно.
        """
        self.weights = [0.0] * self.feature_count
        self.w0 = 0.0
        self.history = []
        self.iterations_done = 0

        n = len(self.norm_data)
        e_counter = 0
        iteration = 0
        errors_this_pass = 0

        while e_counter < n and iteration < max_iterations:
            errors_this_pass = 0
            e_counter = 0

            for obj in self.norm_data:
                d12 = sum(self.weights[i] * obj.features[i] for i in range(self.feature_count)) + self.w0

                correct = (obj.class_label == 1 and d12 > 0) or (obj.class_label == 2 and d12 < 0)

                if correct:
                    e_counter += 1
                else:
                    errors_this_pass += 1
                    if obj.class_label == 1 and d12 <= 0:
                        for i in range(self.feature_count):
                            self.weights[i] += obj.features[i]
                        self.w0 += 1.0
                    elif obj.class_label == 2 and d12 >= 0:
                        for i in range(self.feature_count):
                            self.weights[i] -= obj.features[i]
                        self.w0 -= 1.0

            self.history.append(errors_this_pass)
            iteration += 1

        self.iterations_done = iteration

    def predict_perceptron(self, features_norm: List[float]) -> dict:
        d12 = sum(self.weights[i] * features_norm[i] for i in range(self.feature_count)) + self.w0
        predicted = 1 if d12 > 0 else 2
        equation_parts = [f"{round(w, 3)}*X{i+1}" for i, w in enumerate(self.weights)]
        equation = "D12 = " + " + ".join(equation_parts) + f" + ({round(self.w0, 3)})"
        return {
            "predicted_class": predicted,
            "d12": round(d12, 4),
            "equation": equation,
        }

    # ---------- ДИАГНОСТИКА ----------

    def get_prototypes_info(self) -> dict:
        return {
            "class_1": [round(v, 4) for v in self.prototypes[1]],
            "class_2": [round(v, 4) for v in self.prototypes[2]],
        }

    def get_training_info(self) -> dict:
        return {
            "iterations": self.iterations_done,
            "history": self.history,
            "weights": [round(w, 4) for w in self.weights],
            "w0": round(self.w0, 4),
            "max_values": self.max_values,
        }