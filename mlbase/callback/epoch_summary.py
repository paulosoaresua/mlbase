from mlbase.callback.progress_check import Callback
from typing import Dict, List, Any


class EpochSummary(Callback):

    def __init__(self, display_measures: List[str] = None, precision: int = 3, ):
        super().__init__('epoch', 1)
        self._measures = None
        self._display_measures = display_measures
        self._precision = precision

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        if self._measures is None:
            self._measures = logs.copy()
        else:
            for key in self._measures:
                self._measures[key] += logs[key]

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        for key in self._measures:
            self._measures[key] /= self._num_batches
        self._print_summary()

    def _print_summary(self):
        summary = f"[Epoch {self._epoch}]"
        if self._display_measures is not None:
            for measure in self._display_measures:
                if measure in self._measures:
                    summary += f" {measure}: {self._measures[measure]:.{self._precision}f}"
        print(summary)

