import torch
from .op import PerfOP
from .attention_base import PerfAttentionBase


class PerfNaiveAttention(PerfAttentionBase):
    def __init__(self, b, a, s, h, bias, p, op_index):
        if bias == 0:
            print('bias should not be zero')
            raise
        super(PerfNaiveAttention, self).__init__(b, a, s, h, bias, p, op_index)

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b * self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)
        self.input2 = torch.rand((self.b * self.a, self.h, self.s), device=cuda, dtype=torch.bfloat16)
        self.bias = torch.rand((self.b * self.a , 1, self.s), device=cuda, dtype=torch.bfloat16)
        self.mask = torch.rand((1, self.s, self.s), device=cuda) < 0.9
        self.input3 = torch.rand((self.b * self.a, self.s, self.h), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        tmp = torch.baddbmm(self.bias, self.input1, self.input2, beta=1, alpha=(1.0/0.1))
        # current setting bf16->fp32
        tmp = tmp.float()
        # scale
        tmp = tmp * 5.6
        # fille mask
        tmp.masked_fill_(self.mask, torch.finfo(tmp.dtype).min)
        # use torch softmax to obtain prob
        probs = torch.nn.Softmax(dim=-1)(tmp)

        tmp = tmp.bfloat16()
        if self.p != 0:
            tmp = torch.nn.Dropout(0.3)(tmp)
        out = torch.bmm(tmp, self.input3)

