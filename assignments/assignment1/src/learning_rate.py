class LearningRate():
    """Decreases the gradient by fixed learning rate."""

    def __init__(self, lr=10e-6):
        self.lr = lr

    def compute(self, dw):
        return self.lr * dw

    def __str__(self):
        return "Constant(lr=%s)" % self.lr


class Decay(LearningRate):
    """Decreaces the gradient each iteration at a decay rate."""

    def __init__(self, lr=10e-3, b=10e-4):
        super().__init__(lr)
        self.b = b
        self.decay_step = 1

    def compute(self, dw):
        decay = self.lr / (1 + self.b * self.decay_step)
        self.decay_step += 1
        return decay * dw

    def __str__(self):
        return "Decay(lr={},b={})".format(self.lr, self.b)


class Momentum(LearningRate):
    """GD with momentum. In this case, the grad gives information about the
    acceleration and the momentum gives the velocity."""

    def __init__(self, lr=10e-3, b=0.9):
        super().__init__(lr)
        self.b = b
        self.momentum = 0

    def compute(self, dw):
        self.momentum = (self.momentum + dw) / 2
        grad = self.b * self.momentum + (1 - self.b) * dw
        return self.lr * grad

    def __str__(self):
        return "Momentum(lr={},b={})".format(self.lr, self.b)
