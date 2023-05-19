import os
import subprocess
import torch
import numpy as np
import datetime
import statistics
import torch.nn.functional as F
import datetime
from pytz import timezone

import pandas as pd

import sys
import os

from .op import PerfOP

#bias_add_batch_matmul
class PerfBaddBmm(PerfOP):
    def __init__(self, b, m, k, n):
        super(PerfBaddBmm, self).__init__()
        self.b = b
        self.m = m
        self.k = k
        self.n = n
        self.input1 = None
        self.input2 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.m, self.k), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.b, self.k, self.n), device=cuda, dtype=torch.bfloat16)
        self.bias = torch.rand((self.b, 1, self.n), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        output = torch.baddbmm(self.bias, self.input1, self.input2, beta=1, alpha=(1.0/0.1))
