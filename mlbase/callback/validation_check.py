from mlbase.callback.callback import Callback
from typing import Dict, Any
from mlbase.common.exceptions import NotImplementedError


class ValidationCheck(Callback):
    """
    Stub to compute performance metrics (e.g. accuracy) over a particular dataset.
    Specific callbacks have to be created as a child of this one
    """

    def __init__(self, data: Any, update_frequency_type: str = 'batch', update_frequency: int = 100):
        super().__init__(update_frequency_type, update_frequency)
        self._data = data

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        raise NotImplementedError

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        raise NotImplementedError
