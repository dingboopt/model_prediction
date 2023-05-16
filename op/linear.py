import os
import subprocess
import torch
import numpy as np
import datetime
import statistics
import torch.nn.functional as F
import datetime
from pytz import timezone

import pandas as pd

import sys
import os

def perf_linear(dev, m, k, n):
    with torch.cuda.device(dev):
        cuda = torch.device(f'cuda:{dev}')
        input1 = torch.rand((m, k), device=cuda, dtype=torch.bfloat16)
        input2 = torch.rand((n, k), device=cuda, dtype=torch.bfloat16)

        # warmup
        for _ in range(100):
            xx = torch.rand((m, k), device=cuda, dtype=torch.bfloat16)
            yy = torch.rand((n, k), device=cuda, dtype=torch.bfloat16)
            zz = F.linear(xx, yy)
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
            z = F.linear(input1, input2)
            torch.cuda.synchronize()
            end_times[i] = datetime.datetime.now()
            end_events[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_mean = statistics.mean(times)
        mannul_meaured_times = [(end-start).total_seconds() * 1000 for start,end in zip(start_times, end_times)]
        mannul_meaured_times_mean = statistics.mean(mannul_meaured_times)
        tflops = 2 * m * n * k / 10**12
        Tflops = tflops / times_mean * 1000
        Tflops_mannul = tflops / mannul_meaured_times_mean * 1000
        now_time = datetime.datetime.now(timezone('Asia/Shanghai'))
        d = {'M': [m], 'K':[k], 'N':[n], 'tlops':[tflops], 'SOL':[Tflops/312], 'Mannul_SQL':[Tflops_mannul/312], 'TimeStamp':[now_time.strftime("%Y-%m-%d %H:%M:%S")]}
        df = pd.DataFrame(data=d)
        df = pd.DataFrame(data=d)
        output_path=f'linear_result.csv'
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        print(df)
        print(tflops)
        print(mannul_meaured_times)
        # milisecond
        return mannul_meaured_times_mean
