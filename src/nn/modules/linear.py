import numpy as np
from src.nn.parameter import Parameter
from src.nn.module import Module

class Linear(Module):
    """
    Атрибуты:
    ---------
    in_features: int
        Размерность входного вектора.
    out_features: int
        Размерность выходного вектора.
    bias: bool
        Используется ли вектор смещений.
    W: Parameter
        Матрица весов слоя.
    b: Parameter or None
        Вектор смещений или None, если bias=False.
    """

    def __init__(self, in_features, out_features, bias=True):

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.W = Parameter((in_features, out_features))._init_params("kaiming")
        if self.bias:
            self.b = Parameter(out_features)
        else:
            self.b = None

    def forward(self, x):
        """
        Параметры:
        ----------
        x: np.ndarray, форма (batch_size, in_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, out_features)
            Результат применения Linear к входным данным.
        """
        y = np.dot(x, self.W.data)
        if self.bias:
            y += self.b.data
        self.x = x
        return y

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output: np.ndarray, форма (batch_size, out_features)
            Градиент функции ошибки по выходу Linear.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, in_features)
            Градиент функции ошибки по входу Linear.
        """
        self.W.grad += np.dot(self.x.T, grad_output)
        if self.bias:
            self.b.grad += grad_output.sum(axis=0)
        return np.dot(grad_output, self.W.data.T)

    def parameters(self):
        """
        Возвращает параметры модели.

        Возвращает:
        -----------
        tuple[Parameter]
            Кортеж, содержащий параметры.
        """
        if self.bias:
            return (self.W, self.b)
        else:
            return (self.W,)

    def zero_grad(self):
        """Обнуляет накопленные градиенты модели."""
        self.W.grad = np.zeros_like(self.W.data)
        if self.bias:
            self.b.grad = np.zeros_like(self.W.data)

    def __repr__(self):
        """Строковое представление слоя Linear."""
        return f"Linear({self.in_features}, {self.out_features}, bias={self.bias})"
