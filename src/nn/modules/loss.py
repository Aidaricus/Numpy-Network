import numpy as np

class Loss:
    """
    Класс-контейнер для значения функции ошибки и её градиента.

    Используется для хранения значения ошибки, градиента и ссылки на модель,
    чтобы можно было выполнить обратное распространение ошибки.

    Атрибуты:
    ----------
    loss: float
        Значение функции ошибки.
    grad: np.ndarray
        Градиент ошибки по выходу модели.
    model: объект модели
        Ссылка на модель, чьи параметры нужно обновить.
    """

    def __init__(self, loss, grad, model):
        self.loss = loss
        self.grad = grad
        self.model = model

    def item(self):
        """
        Возвращает значение ошибки как скаляр.

        Возвращает:
        -----------
        float
            Значение функции ошибки.
        """
        return self.loss

    def backward(self):
        """
        Запускает обратное распространение ошибки,
        передавая градиенты в модель.
        """
        self.model._compute_gradients(self.grad)

    def __repr__(self):
        """Возвращает строковое представление ошибки."""
        return str(self.loss)


def CrossEntropyLoss(pred, target):
    """
    Параметры:
    ----------
    pred: Tensor
        Логиты модели, форма (batch_size, num_classes).
    target: np.ndarray, форма (batch_size,)
        Истинные классы в виде одномерного массива.

    Возвращает:
    -----------
    Loss
        Контейнер с ошибкой и градиентом.
    """
    logits = pred.array
    model = pred.model

    # softmax
    exp_pred = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

    batch_size = logits.shape[0]
    correct_log_probs = -np.log(probs[np.arange(batch_size), target])
    loss = np.sum(correct_log_probs) / batch_size

    # Вычисляем градиент
    grad = probs
    grad[np.arange(batch_size), target] -= 1
    grad /= batch_size

    return Loss(loss, grad, model)