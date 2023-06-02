import torch
from .op import PerfOP
from .attention_base import PerfAttentionBase
try:
# Lets explore the speed of each of the 3 implementations
    from torch.backends.cuda import sdp_kernel, SDPBackend
except:
    print('not support spd(scaled dot product)')
import torch.nn as nn
import torch.nn.functional as F

class PerfSDPAttention(PerfAttentionBase):
    def __init__(self, b, a, s, h, bias, p, op_index):
        super(PerfSDPAttention, self).__init__(b, a, s, h, bias, p, op_index)
        # Helpful arguments mapper
        backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {
                "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }

        opt_list = [SDPBackend.MATH, SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        self.opt = backend_map[opt_list[op_index]]
        if self.bias !=0:
            print('SDP currently does not support bias')
            raise
        if self.p !=0 :
            self.p = 0.1
        else:
            self.p = 0

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input3 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        with sdp_kernel(**self.opt):
            F.scaled_dot_product_attention(self.input1, self.input2, self.input3, dropout_p=self.p, is_causal=True)
