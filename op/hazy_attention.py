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
    def __init__(self, b, a, s, h, bias, p, op_index, both_fw_bw):
        super(PerfHazyAttention, self).__init__(b, a, s, h, bias, p, op_index, both_fw_bw)
        if self.p != 0:
            self.p = 0.1
        else:
            self.p = 0
        if self.op_index == 0:
            if bias !=0 :
                print('cutlas does not support bias {bias}')
                raise
            self.op = FlashAttention(softmax_scale = 1.4, attention_dropout=self.p)
        else:

            self.op = flash_attn_func
        



    def prepare_data(self):
        cuda = torch.cuda.current_device()
        if self.op_index == 0:
            self.input1 = torch.rand((self.b , self.s,3,  self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
        else:
            self.input1 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
            self.input2 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
            
            self.input3 = torch.rand((self.b , self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
            self.input4 = torch.rand((self.b , self.a, self.s, self.s), device=cuda, dtype=torch.bfloat16)
            self.input4 = torch.rand((1, self.a, self.s, self.s), device=cuda, dtype=torch.bfloat16)
    def run_kernel(self):
        if self.op_index == 0:
            #y=self.op.apply(self.input1, key_padding_mask=None, causal=True, cu_seqlens=None, max_s=None, need_weights=False)
            y=self.op(self.input1,                  None,        True,            None,       None,              False)[0]
        else:
            if self.bias !=0:
                y=self.op(self.input1, self.input2, self.input3, self.input4, True, 1.4, self.p)
            else:
                y=self.op(self.input1, self.input2, self.input3, None, True, 1.4, self.p)
        if self.both_fw_bw:
            y[0][0][0][0].backward()
    
