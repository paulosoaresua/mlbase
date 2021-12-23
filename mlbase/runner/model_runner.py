import torch
import torch.nn as nn
from mlbase.model.base_model import BaseModel
from typing import List
from mlbase.callback.callback import Callback
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

        for callback in callbacks:
            callback.on_training_begin(len(training_data_loader), True)

        logs = {}
        for epoch in range(self._initial_epoch, epochs):
            for callback in callbacks:
                callback.on_train_epoch_begin(epoch, True)

            for batch, data in enumerate(training_data_loader):
                for callback in callbacks:
                    callback.on_train_batch_begin(batch, True)

                loss = self._model.calculate_loss(data)

                self._optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 5.0)
                self._optimizer.step()

                # Other callbacks might overwrite the model internal log if they call the
                # calculate_loss function again (e.g. EarlyStopping)
                logs.update(self._model.log_keys.copy())
                for callback in callbacks:
                    callback.on_train_batch_end(batch, logs, True)

            for callback in callbacks:
                callback.on_train_epoch_end(epoch, logs, True)

            if self._model.stop_training:
                break

        self._model.eval()
