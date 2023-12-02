import numpy as np
from agents.Network.serialize import convOutputs
from problems.BaseProblem import BaseProblem
from environments.Py2048 import Py2048 as Py2048env
from PyFixedReps.BaseRepresentation import BaseRepresentation

import torch.nn as nn

class IdentityRep(BaseRepresentation):
    def encode(self, s, a=None):
        return s

class FlattenRep(BaseRepresentation):
    def encode(self, s):
        return s.flatten()

class Py2048(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = Py2048env(self.run)
        self.actions = 4

        self.rep = FlattenRep()

        # where features stands for "channels" here
        self.features = 8
        self.gamma = 0.99

        self.max_episode_steps = -1

        self.model = nn.Sequential()

        # conv model
        # self.model.add_module('conv-1', nn.Conv2d(11, 64, 2, 1))
        # self.model.add_module('bn-1', nn.BatchNorm2d(64))
        # self.model.add_module('relu-1', nn.ReLU())
        # self.model.add_module('flatten', nn.Flatten())

        # inputs = convOutputs(4, 64, 2, 1)
        # self.model.add_module('dense-3', nn.Linear(inputs, 64))
        # self.model.add_module('bn-3', nn.BatchNorm1d(64))
        # self.model.add_module('relu-3', nn.ReLU())
        # self.model.add_module('dense-4', nn.Linear(64, 8))
        # self.model.add_module('bn-4', nn.BatchNorm1d(8))
        # self.model.add_module('relu-4', nn.ReLU())

        # dense model
        self.model.add_module('dense-1', nn.Linear(15 * 4 * 4, 64))
        # self.model.add_module('bn-1', nn.BatchNorm1d(64))
        self.model.add_module('relu-1', nn.ReLU())
        self.model.add_module('dense-2', nn.Linear(64, 8))
        # self.model.add_module('bn-2', nn.BatchNorm1d(8))
        self.model.add_module('relu-2', nn.ReLU())

        # linear model
        # self.features = 15 * 4 * 4

        print(self.model)
