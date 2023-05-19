from .op import PerfOP
import torch


class PerfBiasDropoutAddFused(PerfOP):
    def __init__(self, b, h, s):
        super(PerfBiasDropoutAddFused, self).__init__()
        self.b = b
        self.h = h
        self.s = s
        self.input1 = None
        self.input2 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.s, self.b, self.h), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.s, self.b, self.h), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        # add bias then dropout
        out = torch.nn.functional.dropout(self.input1 + self.input2, p=0.6, training=True)
        # residual
        out = out + self.input2
