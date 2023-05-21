from collections import deque

import numpy as np


class MovingAverageMeter:
    def __init__(self, alpha):
        self.avg = None
        self.alpha = alpha

    def update(self, value):
        if self.avg is None:
            self.avg = value
            return
        self.avg = (1 - self.alpha) * self.avg + self.alpha * value

    def reset(self):
        self.avg = None

class WindowAverageMeter:
    def __init__(self, window_size=10):
        self.d = deque(maxlen=window_size)

    def update(self, value):
        self.d.append(value)

    @property
    def avg(self):
        return np.mean(self.d, axis=0)

    def reset(self):
        self.d.clear()
