from typing import Dict, Any


class Callback:
    def __init__(self, update_frequency_type: str = 'batch', update_frequency: int = 100):
        self.update_frequency_type = update_frequency_type
        self.update_frequency = update_frequency

        self._num_batches = 0
        self._epoch = 0
        self._batch = 0
        self._step = 0

    def on_training_begin(self, num_batches: int, train: bool):
        self._num_batches = num_batches
        self._epoch = 0
        self._batch = 0
        self._step = 0
        self._on_training_begin(train)

    def on_train_batch_begin(self, batch: int, train: bool):
        self._batch = batch
        if self.update_frequency_type == 'batch' and batch % self.update_frequency == 0:
            self._step = batch / self.update_frequency + self._epoch * (self._num_batches + 1)
            self._on_train_batch_begin(batch, train)

    def on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        if self.update_frequency_type == 'batch' and batch % self.update_frequency == 0:
            self._on_train_batch_end(batch, logs, train)

    def on_train_epoch_begin(self, epoch: int, train: bool):
        self._epoch = epoch
        if self.update_frequency_type == 'epoch' and epoch % self.update_frequency == 0:
            self._step = self._epoch
            self._on_train_epoch_begin(epoch, train)

    def on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        if self.update_frequency_type == 'epoch' and epoch % self.update_frequency == 0:
            self._on_train_epoch_end(epoch, logs, train)

    # To be implemented by the subclasses
    def _on_training_begin(self, train: bool):
        pass

    def _on_train_batch_begin(self, batch: int, train: bool):
        pass

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        pass

    def _on_train_epoch_begin(self, epoch: int, train: bool):
        pass

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        pass