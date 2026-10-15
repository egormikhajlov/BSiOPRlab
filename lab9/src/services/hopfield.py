import random
from typing import List, Tuple

from src.domain.models import Pattern, NoiseTest


class HopfieldNetwork:
    """
    Сеть Хопфилда для бинарных образов.
    - Образы кодируются биполярно (1 и -1).
    - Обучение по правилу Хебба.
    - Восстановление — синхронное (все нейроны обновляются одновременно).
    """

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self.n = len(patterns[0].matrix) * len(patterns[0].matrix[0])  # 100
        self.size = (len(patterns[0].matrix), len(patterns[0].matrix[0]))  # (10, 10)
        self.weights: List[List[int]] = []
        self._train()

    # ---------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------

    def _matrix_to_vector(self, matrix: List[List[int]]) -> List[int]:
        """Преобразует 10x10 в вектор длиной 100 (биполярный: 0 -> -1)."""
        vec = []
        for row in matrix:
            for v in row:
                vec.append(1 if v == 1 else -1)
        return vec

    def _vector_to_matrix(self, vec: List[int]) -> List[List[int]]:
        """Обратное преобразование: вектор 100 -> 10x10 (бинарный)."""
        rows, cols = self.size
        matrix = []
        idx = 0
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(1 if vec[idx] > 0 else 0)
                idx += 1
            matrix.append(row)
        return matrix

    # ---------- ОБУЧЕНИЕ ----------

    def _train(self):
        """
        Правило Хебба:
            w_ij = sum_k (a_k^i * a_k^j)   при i != j
            w_ii = 0
        """
        self.weights = [[0] * self.n for _ in range(self.n)]

        for pattern in self.patterns:
            vec = self._matrix_to_vector(pattern.matrix)
            for i in range(self.n):
                for j in range(self.n):
                    if i != j:
                        self.weights[i][j] += vec[i] * vec[j]

    # ---------- ВОССТАНОВЛЕНИЕ ----------

    def recall(self, vec: List[int], max_iterations: int = 50) -> List[int]:
        """
        Восстановление образа.
        Синхронное обновление: на каждом шаге все нейроны обновляются одновременно.
        Возвращает стабильный вектор.
        """
        current = vec[:]

        for _ in range(max_iterations):
            new = [0] * self.n
            for i in range(self.n):
                s = sum(self.weights[i][j] * current[j] for j in range(self.n))
                new[i] = 1 if s > 0 else -1

            if new == current:
                break
            current = new

        return current

    def recall_matrix(self, matrix: List[List[int]]) -> Tuple[List[List[int]], bool]:
        """
        Восстановление из бинарной матрицы.
        Возвращает (восстановленная матрица, был ли это один из эталонов).
        """
        vec = self._matrix_to_vector(matrix)
        result_vec = self.recall(vec)
        result_matrix = self._vector_to_matrix(result_vec)

        # Проверка, совпадает ли результат с каким-либо эталоном
        matched = False
        for pattern in self.patterns:
            if pattern.matrix == result_matrix:
                matched = True
                break

        return result_matrix, matched

    # ---------- ЗАШУМЛЕНИЕ ----------

    def add_noise(self, matrix: List[List[int]], percent: int, rng: random.Random) -> List[List[int]]:
        """
        Инвертирует случайные пиксели. percent — от 0 до 100.
        """
        rows, cols = self.size
        total = rows * cols
        n_flip = int(round(total * percent / 100.0))

        flat = [matrix[i][j] for i in range(rows) for j in range(cols)]
        indices = list(range(total))
        rng.shuffle(indices)

        for k in range(n_flip):
            idx = indices[k]
            flat[idx] = 1 - flat[idx]

        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                row.append(flat[i * cols + j])
            result.append(row)
        return result

    # ---------- ТЕСТИРОВАНИЕ ----------

    def run_tests(self, noise_levels: List[int], trials_per_level: int = 10, seed: int = 42) -> List[NoiseTest]:
        """
        Прогоняет серию тестов: для каждого уровня зашумления и каждой буквы
        выполняет trials_per_level попыток восстановления.
        Возвращает список результатов.
        """
        rng = random.Random(seed)
        results: List[NoiseTest] = []

        for pattern in self.patterns:
            for noise in noise_levels:
                success = 0
                for _ in range(trials_per_level):
                    noisy = self.add_noise(pattern.matrix, noise, rng)
                    restored, _ = self.recall_matrix(noisy)
                    if restored == pattern.matrix:
                        success += 1
                results.append(NoiseTest(
                    noise_percent=noise,
                    pattern_name=pattern.name,
                    trials=trials_per_level,
                    success=success,
                    accuracy=success / trials_per_level
                ))

        return results

    # ---------- ДИАГНОСТИКА ----------

    def get_weights_matrix(self) -> List[List[int]]:
        return self.weights