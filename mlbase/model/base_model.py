import torch
import torch.nn as nn
from mlbase.common.exceptions import NotImplementedError
import os
from typing import Any
import copy


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stop_training = False
        self.log_keys = {}
        self._stashed_parameters = None

    def calculate_loss(self, data: Any):
        raise NotImplementedError

    def save(self, out_dir: str):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Model's state
        torch.save(self.state_dict(), '{}/parameters.pt'.format(out_dir))

    def load(self, in_dir: str):
        # Load the pre-trained weights
        self.load_state_dict(torch.load('{}/parameters.pt'.format(in_dir)))

    def stash_parameters(self):
        self._stashed_parameters = copy.deepcopy(self.state_dict())

    def pop_stashed_parameters(self):
        if self._stashed_parameters is not None:
            self.load_state_dict(self._stashed_parameters)
            self._stashed_parameters = None

    def _log_measure(self, measure: str, value: Any):
        mode = 'train' if self.training else 'eval'
        self.log_keys[f"{mode}/{measure}"] = value


