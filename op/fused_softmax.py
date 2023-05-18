from .op import PerfOP
import torch


class PerfFusedScaleMaskSoftmax(PerfOP):
    def __init__(self, b, h, s):
        super(PerfFusedScaleMaskSoftmax, self).__init__()
        self.b = b
        self.h = h
        self.s = s
        self.input1 = None
        self.input2 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.h, self.s, self.s), device=cuda, dtype=torch.bfloat16)
        self.input2 = None
        self.mask = torch.rand((1, 1, self.s, self.s), device=cuda) < 0.9

    def run_kernel(self):

        # current setting bf16->fp32
        input = self.input1.float()
        # scale
        input = input * 5.6
        # fille mask
        input.masked_fill_(self.mask, torch.finfo(input.dtype).min)
        # use torch softmax to obtain prob
        probs = torch.nn.Softmax(dim=-1)(input)

        self.input1 = input.bfloat16()
