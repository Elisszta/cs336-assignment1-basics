from typing import cast

import torch
from torch import Tensor
import torch.nn as nn
import einx
import math
from jaxtyping import Float, Int, Bool


class CrossEntropyLoss(nn.Module):
    1
