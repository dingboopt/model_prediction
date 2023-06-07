import torch
from .op import PerfOP


class PerfAttentionBase(PerfOP):
    def __init__(self, b, a, s, h, bias=0, p=0, op_index=0, both_fw_bw=0):
        super(PerfAttentionBase, self).__init__()
        # micro batch
        self.b = b
        # AttentionBase heads
        self.a = a
        # sequence
        self.s = s
        # hidden size
        self.h = h
        # bias or no bias
        self.bias = bias
        # dropout or no dropout
        self.p = p
        # implementation index
        self.op_index = op_index
        self.both_fw_bw = both_fw_bw

    def get_flops(self):
        # omit softmax and dropout etc. only qk +qk*v
        flops = self.b * self.a * self.s * self.s *self.h * 4
        return flops*3 if self.both_fw_bw  else flops

    def get_config(self):    
        result = {'batch': self.b,'head':self.a, 'seq':self.s, 'hidden':self.h}
        return result
