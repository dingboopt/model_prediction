import torch
from .op import PerfOP

class PerfBmm(PerfOP):
    def __init__(self, b, m, k, n):
        super(PerfBmm, self).__init__()
        self.b = b
        self.m = m
        self.k = k
        self.n = n
        self.input1 = None
        self.input2 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.m, self.k), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.b,self.k, self.n), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        out = torch.bmm(self.input1, self.input2)



