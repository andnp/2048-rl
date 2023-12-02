from typing import Any, Dict, List
import torch
import torch.nn as nn
from utils.torch import device

# if there not anymore convolutional layers, we'll want to add a flatten layer first
def isLastConv(layer_defs: List[Any], i: int):
    t = layer_defs[i]['type']

    n = None
    if i < len(layer_defs) - 1:
        n = layer_defs[i + 1]['type']

    return t == 'conv' and n != 'conv'

# This is a pretty opinionated deserializer for taking a json description and generating
# an instance of a pytorch neural network. By being opinionated, we can reduce the number
# of options that need to be specified in the json; however, we tradeoff generalizability.
# NOTE: just be aware of some of the decisions being made in this module.
class Network(nn.Module):
    def __init__(self, model: nn.Sequential, features: int, outputs: int, seed: int):
        super(Network, self).__init__()
        self.outputs = outputs

        # there's no way to seed a single random call from pytorch
        # so instead, just reset the global seed again.
        torch.manual_seed(seed)

        self.model = model

        # from some of the heads (e.g. for TDRC).
        self.features = features
        self.output = nn.Linear(features, outputs)
        self.output_layers = [self.output]
        self.output_grads = [True]

    # add a new head to the network. Can enable/disable the gradients from this head being passed to feature layers
    def addOutput(self, outputs: int, grad: bool = True, bias: bool = True, initial_value: float = None):
        layer = nn.Linear(self.features, outputs, bias=bias).to(device)

        if initial_value is None:
            nn.init.xavier_uniform_(layer.weight)

        else:
            nn.init.constant_(layer.weight, initial_value)

        if bias:
            nn.init.zeros_(layer.bias)

        self.output_layers.append(layer)
        self.output_grads.append(grad)

        num = len(self.output_layers)
        self.add_module(f'output-{num}', layer)

        return layer

    # take inputs and returns a list of outputs (one for each head)
    # for consistency, always returns a list even for single-headed networks.
    def forward(self, x: torch.Tensor):
        x = self.model(x)
        outs = []
        for layer, grad in zip(self.output_layers, self.output_grads):
            if grad:
                outs.append(layer(x))
            else:
                outs.append(layer(x.detach()))

        return outs

# used for copying to target networks.
def cloneNetworkWeights(fromNet: Network, toNet: Network):
    toNet.load_state_dict(fromNet.state_dict())
