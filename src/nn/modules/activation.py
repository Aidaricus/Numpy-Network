import numpy as np
from src.nn.module import Module

class ReLU(Module):

    def forward(self, x):
        """
        Параметры:
        ----------
        x: np.ndarray, форма (batch_size, num_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Результат применения ReLU к входным данным.
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output: np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по выходу ReLU.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по входу ReLU.
        """
        grad_input = grad_output * (self.input >= 0).astype(self.input.dtype)
        return grad_input

    def __repr__(self):
        """Строковое представление слоя ReLU."""
        return f"ReLU()"

class Sigmoid(Module):

    def forward(self, x):
        """
        Параметры:
        ----------
        x : np.ndarray, форма (batch_size, num_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Результат применения сигмоида к входным данным.
        """
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output : np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по выходу сигмоида.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по входу сигмоида.
        """
        return self.output * (1 - self.output) * grad_output

    def __repr__(self):
        """Строковое представление слоя Sigmoid."""
        return f"Sigmoid()"


class Tanh(Module):

    def forward(self, x):
        """
        Параметры:
        ----------
        x : np.ndarray, форма (batch_size, num_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Результат применения Tanh к входным данным.
        """
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output : np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по выходу Tanh.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по входу Tanh.
        """
        return (1 - self.output ** 2) * grad_output


    def __repr__(self):
        """Строковое представление слоя Tanh."""
        return f"Tanh()"