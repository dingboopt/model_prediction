import os
import subprocess
import torch
import numpy as np
import datetime
import statistics
from megatron.model.fused_bias_gelu import bias_gelu_impl
import torch.nn.functional as F
import datetime
from pytz import timezone

import pandas as pd

import sys
import os

def perf_gelu(dev, input, bias):
    with torch.cuda.device(dev):
        cuda = torch.device(f'cuda:{dev}')
        input1 = torch.rand(input, device=cuda, dtype=torch.bfloat16)
        input2 = torch.rand(bias, device=cuda, dtype=torch.bfloat16)

        # warmup
        for _ in range(100):
            xx = torch.rand(input, device=cuda, dtype=torch.bfloat16)
            yy = torch.rand(bias, device=cuda, dtype=torch.bfloat16)
            zz = bias_gelu_impl(xx, yy)
        torch.cuda.synchronize()
        steps = 10
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        start_times = [0 for _ in range(steps)]
        end_times = [0 for _ in range(steps)]
        for i in range(steps):
            torch.cuda.synchronize()
            start_events[i].record()
            start_times[i] = datetime.datetime.now()
            z = bias_gelu_impl(input1, input2)
            torch.cuda.synchronize()
            end_times[i] = datetime.datetime.now()
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_mean = statistics.mean(times)
        mannul_meaured_times = [(end-start).total_seconds() * 1000 for start,end in zip(start_times, end_times)]
        mannul_meaured_times_mean = statistics.mean(mannul_meaured_times)

        # milisecond
        return mannul_meaured_times_mean
