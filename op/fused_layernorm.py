import torch
from megatron.model.fused_layer_norm import FusedLayerNormAffineFunction
from megatron.model.fused_layer_norm import  MixedFusedLayerNorm
import torch.nn.functional as F
import datetime
import statistics

from .op import PerfOP

class PerfFusedLayerNorm(PerfOP):
    def __init__(self, b, h, s):
        super(PerfFusedLayerNorm, self).__init__()
        self.b = b
        self.h = h
        self.s = s

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.h), device=cuda, dtype=torch.bfloat16)
        self.input3 = torch.rand((self.h), device=cuda, dtype=torch.bfloat16)
        self.input4 = [self.h]
        self.input5 = 1e-05

    def run_kernel(self):
        F.layer_norm(self.input1, self.input4, self.input2 , self.input3)
