import numpy as np
from src.nn.parameter import Parameter
from src.nn.module import Module

class BatchNorm(Module):
    """
    Атрибуты:
    ----------
    num_features: int
        Количество входных признаков (размерности входного вектора).
    momentum: float, по умолчанию 0.9
        Коэффициент экспоненциального сглаживания для обновления скользящих статистик.
    eps: float, по умолчанию 1e-100
        Малое число для предотвращения деления на ноль.
    gamma: Parameter
        Обучаемый параметр масштабирования.
    beta: Parameter
        Обучаемый параметр сдвига.
    running_mean: np.ndarray
        Скользящее среднее для нормализации данных во время инференса.
    running_var: np.ndarray
        Скользящая дисперсия для нормализации данных во время инференса.
    """

    def __init__(self, num_features, momentum=0.9, eps=1e-100):

        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = Parameter((num_features,))
        self.gamma._init_params(method='ones')
        self.beta = Parameter((num_features,))
        self.beta._init_params(method='zeros')
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
        self.x_centered = None
        self.x_hat = None
        self.var = None

    def train(self):
        """Переводит слой в режим обучения."""
        self.training = True

    def eval(self):
        """Переводит слой в режим инференса."""
        self.training = False

    def forward(self, x):
        """
        Параметры:
        ----------
        x: np.ndarray, форма (batch_size, num_features)
            Входные данные.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Результат применения BatchNorm к входным данным.
        """
        if self.training:
            mu = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.x_centered = x - mu
            self.var = var
            self.x_hat = self.x_centered / np.sqrt(var + self.eps)
            return self.gamma.data * self.x_hat + self.beta.data

        x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma.data * x_hat + self.beta.data

    def zero_grad(self):
        """Обнуляет накопленные градиенты."""
        self.gamma.data = np.zeros_like(self.gamma.data)
        self.beta.data = np.zeros_like(self.beta.data)

    def backward(self, grad_output):
        """
        Параметры:
        ----------
        grad_output: np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по выходу BatchNorm.

        Возвращает:
        -----------
        np.ndarray, форма (batch_size, num_features)
            Градиент функции ошибки по входу BatchNorm.
        """
        batch_size = grad_output.shape[0]

        self.gamma.grad += np.sum(grad_output * self.x_hat, axis=0)
        self.beta.grad += np.sum(grad_output, axis=0)

        dx_hat = grad_output * self.gamma.data
        dvar = np.sum(dx_hat * self.x_centered * -0.5 * (self.var + self.eps) ** (-1.5), axis=0)
        dmu = np.sum(dx_hat * -1 / np.sqrt(self.var + self.eps), axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        dx = dx_hat / np.sqrt(self.var + self.eps) + dvar * 2 * self.x_centered / batch_size + dmu / batch_size

        return dx

    def __repr__(self):
        """Строковое представление слоя Linear."""
        return f"BatchNorm(num_features={self.num_features}, momentum={self.momentum}, eps={self.eps})"

    def parameters(self):
        """
        Возвращает параметры модели.

        Возвращает:
        -----------
        tuple[Parameter]
            Кортеж, содержащий параметры.
        """
        return (self.gamma, self.beta)