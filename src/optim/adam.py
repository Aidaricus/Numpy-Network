import numpy as np

class Adam:
    """
    Атрибуты:
    ----------
    params: iterable
        Итерируемый объект, содержащий параметры модели, которые нужно оптимизировать.
    lr: float, optional, default=3e-4
        Learning rate (скорость обучения).
    beta_1: float, optional, default=0.9
        Коэффициент для оценки первого момента градиентов (среднее).
    beta_2: float, optional, default=0.999
        Коэффициент для оценки второго момента градиентов (нецентрированная дисперсия).
    eps: float, optional, default=1e-8
        Малое число для предотвращения деления на ноль.
    weight_decay: float, optional, default=None
        Коэффициент для L2-регуляризации.
    t: int
        Счетчик шагов оптимизации
    """

    def __init__(self, params, lr=3e-4, beta_1=0.9, beta_2=0.999, eps=1e-8, weight_decay=0):
        self.params = list(params)
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(param.data) for param in self.params]
        self.v = [np.zeros_like(param.data) for param in self.params]

    def zero_grad(self):
        """
        Обнуляет градиенты всех параметров.
        """
        for param in self.params:
            if param is not None:
                param.grad.fill(0)

    def step(self):
        """
        Выполняет один шаг оптимизации Adam.
        """
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            if self.weight_decay != 0:
                param.grad += self.weight_decay * param.data

            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * param.grad

            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (param.grad ** 2)

            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)

            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)