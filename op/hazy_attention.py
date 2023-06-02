import torch
from .op import PerfOP
from .attention_base import PerfAttentionBase
### This is a ref to hazy
from .flash_attention_triton_imp import flash_attn_func

try:
# Lets explore the speed of each of the 3 implementations
    from flash_attn.flash_attention import FlashAttention
except:
    print('not support spd(scaled dot product)')


class PerfHazyAttention(PerfAttentionBase):
    def __init__(self, b, a, s, h, bias, p, op_index):
        super(PerfHazyAttention, self).__init__(b, a, s, h, bias, p, op_index)
        if self.p != 0:
            self.p = 0.1
        else:
            self.p = 0
        if self.op_index == 0:
            self.op = FlashAttention(softmax_scale = 1.4, attention_dropout=self.p).forward
        else:
            self.op = flash_attn_func
        
        if op_index == 0:
            if p !=0 or bias !=0 :
                print('cutlas does not support dropout{p} or bias{bias}')
                raise


    def prepare_data(self):
        cuda = torch.cuda.current_device()
        if self.op_index == 0:
            self.input1 = torch.rand((self.b , self.s,3,  self.a, self.h), device=cuda, dtype=torch.bfloat16)
        else:
            self.input1 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16)
            self.input2 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16)
            self.input3 = torch.rand((1 , self.a , 1, self.s), device=cuda, dtype=torch.bfloat16)
            self.input3 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        if self.op_index == 0:
            self.op(self.input1, key_padding_mask=None, causal=True, cu_seqlens=None, max_s=None, need_weights=False)
        else:
            if self.bias !=0:
                self.op(self.input1, self.input2, self.input3, self.input3, True, self.p)
            else:
                self.op(self.input1, self.input2, self.input3, None, True, self.p)
    
