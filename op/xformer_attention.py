import torch
from .op import PerfOP
import xformers.ops as xops
from .attention_base import PerfAttentionBase

class PerfXformerAttention(PerfAttentionBase):
#Input tensors must be in format [B, M, H, K], where B is the batch size, M the 
#sequence length, H the number of heads, and K the embeding size per head
#If inputs have dimension 3, it is assumed that the dimensions are [B, M, K] and H=1
    def __init__(self, b, a, s, h, bias, p, op_index, both_fw_bw):
        super(PerfXformerAttention, self).__init__(b, a, s, h, bias, p, op_index, both_fw_bw)
        ops = [(xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp),\
        (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp),\
        (xops.fmha.triton.FwOp, xops.fmha.triton.BwOp),\
        (xops.fmha.small_k.FwOp, xops.fmha.small_k.BwOp),\
        ]     
        self.op = ops[self.op_index]

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        #q
        self.input1 = torch.rand((self.b, self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
        #k
        self.input2 = torch.rand((self.b, self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
        #v
        self.input3 = torch.rand((self.b, self.s, self.a, self.h), device=cuda, dtype=torch.bfloat16,requires_grad=True)
        #bias
        self.input4 = torch.rand((self.b*self.a, self.s, self.s), device=cuda, dtype=torch.bfloat16)
        #lse
        self.input5 = torch.rand((self.b,self.a, self.s  ), device=cuda, dtype=torch.float32,requires_grad=True)

    def run_kernel(self):
        # Causal attention
        if self.bias != 0:
            bias = xops.fmha.attn_bias.LowerTriangularMaskWithTensorBias(self.input4) 
        else:
            bias = xops.fmha.attn_bias.LowerTriangularMask()

        if self.p != 0:
            p = 0.1
        else:
            p = 0



        y = xops.memory_efficient_attention(
            self.input1, self.input2, self.input3,
            attn_bias=bias,
            p = p,
            scale=1.4,
            op=self.op
        )
        if self.both_fw_bw:        
            y[0][0][0][0].backward()
        
#        z = xops.memory_efficient_attention_forward_requires_grad(self.input1, self.input2, self.input3,
#            attn_bias=bias,
#            p = p,
#            scale=1.4,
#            op=self.op[0])
#
#        z = xops.memory_efficient_attention_backward(self.input1,self.input1, self.input5, self.input1, self.input2, self.input3,
#            attn_bias=bias,
#            p = p,
#            scale=1.4,
#            op=self.op[1])

