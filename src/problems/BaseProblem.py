import numpy as np
from agents.registry import getAgent
from PyExpUtils.utils.dict import merge

class BaseProblem:
    def __init__(self, exp, idx):
        self.exp = exp
        self.idx = idx

        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']
        self.run = exp.getRun(idx)
        self.seed = self.run

        self.agent = None
        self.model = None
        self.env = None
        self.rep = None
        self.gamma = None

        self.features = 0
        self.actions = 0

    def getEnvironment(self):
        return self.env

    def getRepresentation(self):
        return self.rep

    def getGamma(self):
        return self.gamma

    def getAgent(self, collector):
        Agent = getAgent(self.exp.agent)

        params = merge(self.params, {'gamma': self.getGamma()})
        self.agent = Agent(self.model, self.features, self.actions, params, self.seed, collector)
        return self.agent

    def getSteps(self):
        return self.exp.steps
