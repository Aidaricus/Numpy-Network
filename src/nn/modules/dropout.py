import numpy as np
from src.nn.module import Module

class Dropout(Module):
    """
    Атрибуты:
    ----------
    p: float, по умолчанию 0.5
        Вероятность зануления элемента.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.is_training = True
        self.mask = None

    def train(self):
        """Переводит слой в режим обучения."""
        self.is_training = True

    def eval(self):
        """Переводит слой в режим инференса"""
        self.is_training = False

    def forward(self, x):
        """
        Параметры:
        ----------
        x: np.ndarray, форма (batch_size, num_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Результат применения Dropout к входным данным.
        """
        if self.is_training:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x * self.mask
        else:
            # В режиме инференса Dropout не применяется
            return x * (1 - self.p)

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output: np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по выходу Dropout.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по входу Dropout.
        """
        if self.is_training:
            return grad_output * self.mask
        else:
            return grad_output

    def __repr__(self):
        """Строковое представление слоя Dropout."""
        return f"Dropout(p={self.p})"
