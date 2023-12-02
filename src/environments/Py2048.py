import numpy as np
from Py2048 import Game
from RlGlue import BaseEnvironment

# let 15 be the largest achievable value
def asOneHot(s):
    return (np.arange(15) == s[..., None]).astype(int).transpose(2, 0, 1)

class Py2048(BaseEnvironment):
    def __init__(self, seed):
        self.rng = np.random.RandomState(seed)
        self.env = Game(additive=True, rng=self.rng)
        self.high_block = 1

    def start(self):
        self.high_block = 1
        self.env = Game(additive=True, rng=self.rng)
        return asOneHot(self.env.getBoard())

    def step(self, a):
        t = self.env.takeAction(a)
        b = self.env.getBoard()
        sp = asOneHot(b)

        # add a small per-move penalty to avoid making useless moves
        r = -0.5

        current_high = np.max(b)
        if current_high > self.high_block:
            self.high_block = current_high

        if t:
            r = self.env.getScore()

        if not t:
            print(b, end='\033[F\033[F\033[F')
        else:
            print(b)

        # print(b)

        return (r, sp, t)
