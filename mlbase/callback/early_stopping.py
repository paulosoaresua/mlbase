from mlbase.callback.callback import Callback
from typing import Dict, Any, Callable
from mlbase.model.base_model import BaseModel


class EarlyStopping(Callback):
    """
    Stub to compute performance metrics (e.g. accuracy) over a particular dataset.
    Specific callbacks have to be created as a child of this one
    """

    def __init__(self, model: BaseModel, monitor: str, patience: int, comparator: Callable = lambda x, y: x < y,
                 tolerance: float = 1e-4):
        super().__init__('epoch', 1)
        self._model = model
        self._monitor = monitor
        self._patience = patience
        self._tolerance = tolerance
        self._comparator = comparator
        self._unchanged_epochs = 0
        self._last_measure_value = None

    def _on_training_begin(self, train: bool):
        self._unchanged_epochs = 0

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        if self._last_measure_value is None:
            self._last_measure_value = logs.get(self._monitor, None)
        else:
            curr_measure_value = logs[self._monitor]
            if abs(curr_measure_value - self._last_measure_value) <= self._tolerance:
                self._unchanged_epochs += 1
            else:
                self._unchanged_epochs = 0
                if self._comparator(curr_measure_value, self._last_measure_value):
                    self._model.stash_parameters()

            if self._unchanged_epochs > self._patience:
                self._model.pop_stashed_parameters()
                self._model.stop_training = True

            self._last_measure_value = curr_measure_value
