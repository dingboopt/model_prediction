import torch
from .op import PerfOP
# Lets explore the speed of each of the 3 implementations
from torch.backends.cuda import sdp_kernel, SDPBackend
import torch.nn as nn
import torch.nn.functional as F

class PerfFlashAttention(PerfOP):
    def __init__(self, b, a, s, h):
        super(PerfFlashAttention, self).__init__()
        self.b = b
        self.a = a
        self.s = s
        self.h = h
        self.input1 = None
        self.input2 = None
        self.input3 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input3 = torch.rand((self.b, self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        # Helpful arguments mapper
        backend_map = {
            SDPBackend.MATH: {"enable_math": True, "enable_flash": False, "enable_mem_efficient": False},
            SDPBackend.FLASH_ATTENTION: {"enable_math": False, "enable_flash": True, "enable_mem_efficient": False},
            SDPBackend.EFFICIENT_ATTENTION: {
                "enable_math": False, "enable_flash": False, "enable_mem_efficient": True}
        }

        with sdp_kernel(**backend_map[SDPBackend.FLASH_ATTENTION]):
            F.scaled_dot_product_attention(self.input1, self.input2, self.input3, dropout_p=0.5, is_causal=True)

