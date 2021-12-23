from mlbase.callback.callback import Callback
from typing import Dict, Set, Any


class EpochSummary(Callback):

    def __init__(self):
        super().__init__('epoch', 1)
        self._measures = None

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        if self._measures is None:
            self._measures = logs.copy()
        else:
            for key in self._measures:
                self._measures[key] += logs[key]

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        for key in self._measures:
            self._measures[key] /= self._num_batches

