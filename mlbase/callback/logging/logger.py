from typing import Dict, Any, List
import datetime
from mlbase.callback.callback import Callback
import torch


class Logger(Callback):
    def __init__(self,
                 id: str = None,
                 display_measures: List[str] = None,
                 update_frequency_type: str = 'batch',
                 update_frequency: int = 1):
        super().__init__(update_frequency_type, update_frequency)
        if id is None:
            self.id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        else:
            self.id = id

        self._display_measures = display_measures

        # This will aggregate the measures per batch in case the logger's update frequency type is per epoch
        self._aggregated_measures = None

    def on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        if self._aggregated_measures is None:
            self._aggregated_measures = logs.copy()
        else:
            for key in self._aggregated_measures:
                self._aggregated_measures[key] += logs[key]

        super().on_train_batch_end(batch, logs, train)

    def on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        for key in self._aggregated_measures:
            self._aggregated_measures[key] /= self._num_batches

        super().on_train_epoch_end(epoch, logs, train)

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        self.log_scalars(logs, train)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        scalars = logs.copy()
        scalars.update(self._aggregated_measures)
        self.log_scalars(scalars, train)

    def log_scalars(self, scalars: Dict[str, Any], train: bool):
        if self._display_measures is None:
            for key, value in scalars.items():
                self.log_scalar(key, value, self._step, train)
        else:
            for key in self._display_measures:
                value = scalars.get(key, None)
                if value is not None:
                    self.log_scalar(key, value, self._step, train)

    def log_scalar(self, measure: str, value: Any, step: int, train: bool):
        raise NotImplementedError

    def log_hyper_parameters(self, performance_measures: Dict[str, Any], hyper_parameters: Dict[str, float]):
        raise NotImplementedError

    def log_image(self, measure: str, image: torch.tensor, step: int, train: bool):
        raise NotImplementedError
