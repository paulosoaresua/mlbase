import torch
import torch.nn as nn
from mlbase.model.base_model import BaseModel
from typing import List
from mlbase.callback.callback import Callback
from mlbase.callback.validation_check import ValidationCheck
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np


class ModelRunner:
    def __init__(self, model: BaseModel, optimizer: torch.optim):
        self._model = model
        self._optimizer = optimizer
        self._initial_epoch = 0
        self._initial_batch = 0
        self._random_state_initializers = None

    def reset(self):
        self._initial_epoch = 0
        self._initial_batch = 0
        self._random_state_initializers = None

    def load(self, in_dir: str, save_point: int = None):
        if save_point is None:
            # Retrieve the last saved model in the folder
            pass

        filename = f"{in_dir}/model.{save_point}.pt"
        data_package = torch.load(filename)

        self._model.load_state_dict(data_package['model_state_dict'])
        self._optimizer.load_state_dict(data_package['optimizer_state_dict'])

        random.setstate(data_package['random_state'])
        np.random.set_state(data_package['numpy_random_state'])
        torch.set_rng_state(data_package['torch_random_state'])
        self._initial_epoch = data_package['epoch'] + 1

    def train(self, training_set: Dataset, epochs: int, batch_size: int, callbacks: List[Callback]):
        self._model.train()
        self._model.stop_training = False

        training_data_loader = DataLoader(training_set, batch_size=batch_size)

        # All occurrences of a ValidationCheck callback must be the first ones so that other callbacks
        # have access to the measures computed by the former.
        for i in range(len(callbacks)):
            if isinstance(callbacks[i], ValidationCheck):
                callback = callbacks.pop(i)
                callbacks.insert(0, callback)

        for callback in callbacks:
            callback.on_training_begin(len(training_data_loader), True)

        for epoch in range(self._initial_epoch, epochs):
            for callback in callbacks:
                callback.on_train_epoch_begin(epoch, True)

            for batch, data in enumerate(training_data_loader):
                self._model.log_keys.clear()
                for callback in callbacks:
                    callback.on_train_batch_begin(batch, True)

                loss = self._model.calculate_loss(data)

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._optimizer.step()

                for callback in callbacks:
                    callback.on_train_batch_end(batch, self._model.log_keys, True)

                # Clear the log because if it's preserved and there's a callback per epoch to save the log,
                # it will save the result from the last batch. Callbacks per epoch should be combined with a
                # ValidationCheck or a EpochSummary callback so results can be computed over a full dataset.
                self._model.log_keys.clear()

            for callback in callbacks:
                callback.on_train_epoch_end(epoch, self._model.log_keys, True)

            if self._model.stop_training:
                break

        self._model.eval()
