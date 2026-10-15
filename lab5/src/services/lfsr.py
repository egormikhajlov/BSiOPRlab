class LFSR:
    """
    32-битный регистр сдвига с линейной обратной связью (LFSR).
    Полином №6: X^32 + X^31 + X^30 + X^29 + X^27 + X^25 + X^23 + X^21 +
                X^20 + X^19 + X^13 + X^10 + X^7 + X^6 + X^5 + X^4 + X^3 + X^1 + 1
    """
    # Биты, участвующие в обратной связи (степени X от 0 до 31, без X^32)
    TAPS = [31, 30, 29, 28, 26, 24, 22, 20, 19, 18, 12, 9, 6, 5, 4, 3, 1, 0]

    def __init__(self, seed: int = 0xACE1):
        if seed == 0:
            raise ValueError("Начальное состояние LFSR не может быть нулевым.")
        self.state = seed & 0xFFFFFFFF
        self.mask = 0
        for bit in self.TAPS:
            self.mask |= (1 << bit)

    def next_bit(self) -> int:
        """Возвращает очередной бит (0 или 1) и сдвигает регистр."""
        # Определяем бит обратной связи как XOR всех отводов
        feedback = 0
        for bit in self.TAPS:
            feedback ^= (self.state >> bit) & 1
        # Сдвигаем регистр вправо, в старший бит записываем feedback
        self.state = ((self.state >> 1) | (feedback << 31)) & 0xFFFFFFFF
        return feedback

    def next_uint32(self) -> int:
        """Возвращает 32-битное число, собранное из 32 последовательных битов."""
        value = 0
        for _ in range(32):
            value = (value << 1) | self.next_bit()
        return value

    def next_float(self) -> float:
        """Возвращает псевдослучайное число в диапазоне [0, 1)."""
        return self.next_uint32() / 2**32