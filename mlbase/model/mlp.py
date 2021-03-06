import torch
import torch.nn as nn
from typing import List
from mlbase.common.get_activation import get_activation


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: str, hidden_dims: List[int],
                 activation_on_last_layer: bool = False):
        super(MLP, self).__init__()
        self._in_features = in_features
        self._out_features = out_features
        self._activation = activation
        self._hidden_dims = hidden_dims
        self._activation_on_last_layer = activation_on_last_layer

        self._neural_network = None

        self._build_nn()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self._neural_network(x)

    def _build_nn(self):
        dims = [self._in_features]
        dims.extend(self._hidden_dims)
        dims.append(self._out_features)

        layers = []
        for l in range(1, len(dims)):
            layers.append(nn.Linear(dims[l - 1], dims[l]))

            if self._activation is not None:
                if l < len(dims) - 1 or self._activation_on_last_layer:
                    layers.append(get_activation(self._activation))

        self._neural_network = nn.Sequential(*layers)
