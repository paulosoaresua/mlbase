from mlbase.callback.logging.logger import Logger
from typing import Dict, List, Any


class ProgressCheck(Logger):

    def __init__(self, display_measures: List[str] = None, precision: int = 3, update_frequency_type: str = 'batch',
                 update_frequency: int = 100):
        super().__init__(update_frequency_type=update_frequency_type, update_frequency=update_frequency)
        self._display_measures = display_measures
        self._precision = precision

    def log_scalars(self, scalars: Dict[str, Any], train: bool):
        if self.update_frequency_type == 'batch':
            progress = f"[Epoch {self._epoch} Batch {self._batch}]"
        else:
            progress = f"[Epoch {self._epoch}]"

        if self._display_measures is None:
            for key, value in scalars.items():
                progress += f" {key}: {value:.{self._precision}f}"
        else:
            for key in self._display_measures:
                value = scalars.get(key, None)
                if value is not None:
                    progress += f" {key}: {value:.{self._precision}f}"
        print(progress)
