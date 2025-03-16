import time

class Timer:
    def __init__(self, history: int = 16):
        super().__init__()
        self.index = 0
        self.begin = None
        self.times = [0.0] * history
        self.history = history

    def start(self):
        self.begin = time.time()

    def stop(self):
        if self.begin is None:
            return

        t = time.time()
        elapsed = t - self.begin
        self.begin = t

        self.times[self.index % self.history] = elapsed
        self.index += 1

        return self.elapsed()

    def elapsed(self):
        l = min(self.index, self.history)
        return 0 if l == 0 else sum(self.times[:l]) / l

    def frequency(self):
        e = self.elapsed()
        return 0 if e == 0 else 1.0 / e
