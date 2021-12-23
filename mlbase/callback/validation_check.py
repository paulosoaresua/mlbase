from mlbase.callback.callback import Callback
from typing import Dict, Any
from mlbase.model.base_model import BaseModel
import torch
import random
import numpy as np
from torch.utils.data.dataloader import DataLoader


class ValidationCheck(Callback):
    """
    Stub to compute performance metrics (e.g. accuracy) over a particular dataset.
    Specific callbacks have to be created as a child of this one
    """

    def __init__(self, model: BaseModel, data: Any, update_frequency_type: str = 'epoch', update_frequency: int = 1):
        super().__init__(update_frequency_type, update_frequency)
        self._model = model
        self._data_set = data

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        self._calculate_validation_loss()

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        self._calculate_validation_loss()

    def _calculate_validation_loss(self):
        self._stash_random_state()
        training_mode = self._model.training

        self._model.eval()
        with torch.no_grad():
            # This will compute the losses for the dataset and store the values in a log dictionary in the model
            data_loader = DataLoader(self._data_set, batch_size=len(self._data_set))
            self._model.calculate_loss(next(iter(data_loader)))

            # Set state to its original value
            self._model.train(training_mode)
            self._pop_stashed_random_state()
