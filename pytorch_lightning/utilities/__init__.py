"""General utilities"""
import importlib.util

import numpy
import torch

from pytorch_lightning.utilities.distributed import rank_zero_only, rank_zero_warn, rank_zero_info
from pytorch_lightning.utilities.apply_func import move_data_to_device
from pytorch_lightning.utilities.parsing import AttributeDict


APEX_AVAILABLE = importlib.util.find_spec("apex") is not None
XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
HOROVOD_AVAILABLE = importlib.util.find_spec("horovod") is not None
NATIVE_AMP_AVALAIBLE = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
TORCHTEXT_AVAILABLE = importlib.util.find_spec("torchtext") is not None


FLOAT16_EPSILON = numpy.finfo(numpy.float16).eps
FLOAT32_EPSILON = numpy.finfo(numpy.float32).eps
FLOAT64_EPSILON = numpy.finfo(numpy.float64).eps
