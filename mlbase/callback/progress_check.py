from mlbase.callback.callback import Callback
from typing import Dict, List, Any


class ProgressCheck(Callback):

    def __init__(self, display_measures: List[str] = None, precision: int = 3, update_frequency_type: str = 'batch',
                 update_frequency: int = 100):
        super().__init__(update_frequency_type, update_frequency)
        self._display_measures = display_measures
        self._precision = precision

    def _on_train_batch_end(self, batch: int, logs: Dict[str, Any], train: bool):
        self._print_progress(logs)

    def _on_train_epoch_end(self, epoch: int, logs: Dict[str, Any], train: bool):
        self._print_progress(logs)

    def _print_progress(self, logs: Dict[str, float]):
        if self.update_frequency_type == 'batch':
            progress = f"[Epoch {self._epoch} Batch {self._batch}]"
        else:
            progress = f"[Epoch {self._epoch}]"

        if self._display_measures is not None:
            for measure in self._display_measures:
                if measure in logs:
                    progress += f" {measure}: {logs[measure]:.{self._precision}f}"
        print(progress)
