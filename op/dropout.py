from .op import PerfOP
import torch


class PerfDropout(PerfOP):
    def __init__(self, shape):
        super(PerfDropout, self).__init__()
        self.shape = shape
        self.input1 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand(self.shape, device=cuda, dtype=torch.bfloat16)

    def run_kernel(self):
        probs = torch.nn.Dropout(0.3)(self.input1)
