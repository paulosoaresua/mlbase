from typing import Any, Dict, Callable
from mlbase.callback.logging.logger import Logger
from torch.utils.tensorboard import SummaryWriter
import torch


class TensorBoardLogger(Logger):

    def __init__(self, out_dir: str, id: str = None, update_frequency_type: str = 'batch', update_frequency: int = 1,
                 image_transforms: Dict[str, Callable] = None):
        super().__init__(id=id, update_frequency_type=update_frequency_type, update_frequency=update_frequency)
        self.image_transforms = image_transforms

        self.board_dir = '{}/{}'.format(out_dir, self.id)
        self._writer = SummaryWriter(self.board_dir)

        # This will aggregate the measures per batch in case the logger's update frequency type is per epoch
        self._measures = None

    def log_scalar(self, measure: str, value: Any, step: int, train: bool):
        if isinstance(value, float):
            self._writer.add_scalar(measure, value, step)

    def log_hyper_parameters(self, performance_measures: Dict[str, float], hyper_parameters: Dict[str, float]):
        self._writer.add_hparams(metric_dict=performance_measures, hparam_dict=hyper_parameters)

    def log_image(self, measure: str, image: torch.tensor, step: int, train: bool):
        self._writer.add_image(measure, image, step)

    def __del__(self):
        self._writer.flush()
        self._writer.close()
