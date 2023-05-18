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

def perf_baddbmm(dev, b, m, k, n):
    with torch.cuda.device(dev):
        cuda = torch.device(f'cuda:{dev}')
        print(f'b is {b}, m is {m}, k is {k}, n is {n}')
        input1 = torch.rand((b, m, k), device=cuda, dtype=torch.bfloat16)
        input2 = torch.rand((b, k, n), device=cuda, dtype=torch.bfloat16)
        bias =   torch.rand((b, 1, n), device=cuda, dtype=torch.bfloat16)
        output = torch.rand((b, m, n), device=cuda, dtype=torch.bfloat16)

        # warmup
        for _ in range(100):
            xx = torch.rand((b, m, k), device=cuda, dtype=torch.bfloat16)
            yy = torch.rand((b, k, n), device=cuda, dtype=torch.bfloat16)
            zz = torch.rand((b, m, n), device=cuda, dtype=torch.bfloat16)
            zz = torch.baddbmm(zz, xx, yy, beta=0.0, alpha=(1.0/0.1))
        torch.cuda.synchronize()
        steps = 10
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        start_times = [0 for _ in range(steps)]
        end_times = [0 for _ in range(steps)]



        def trace_handler(prof):
            pass
            #print(prof.key_averages().table(
            #    sort_by="self_cuda_time_total", row_limit=10))
            #torch.profiler.tensorboard_trace_handler('./log/baddbmm4tp')(prof)


        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=2,
                active=2,
                repeat=1),
                with_flops=True,
                on_trace_ready=trace_handler,
                record_shapes=True,
                with_stack=True
            # used when outputting for tensorboard
            ) as p:
            for i in range(steps):
                torch.cuda.synchronize()
                start_events[i].record()
                start_times[i] = datetime.datetime.now()
                output = torch.baddbmm(bias, input1, input2, beta=1, alpha=(1.0/0.1))
                torch.cuda.synchronize()
                end_times[i] = datetime.datetime.now()
                end_events[i].record()



                p.step()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_mean = statistics.mean(times)
        mannul_meaured_times = [(end-start).total_seconds() * 1000 for start,end in zip(start_times, end_times)]
        mannul_meaured_times_mean = statistics.mean(mannul_meaured_times)
        tflops = 2 * b * m * n * k / 10**12
        Tflops = tflops / times_mean * 1000
        Tflops_mannul = tflops / mannul_meaured_times_mean * 1000
        now_time = datetime.datetime.now(timezone('Asia/Shanghai'))
        d = { 'B':[b], 'M': [m], 'K':[k], 'N':[n], 'tlops':[tflops], 'SOL':[Tflops/312], 'Mannul_SQL':[Tflops_mannul/312], 'TimeStamp':[now_time.strftime("%Y-%m-%d %H:%M:%S")]}
        df = pd.DataFrame(data=d)
        df = pd.DataFrame(data=d)
        output_path=f'baddbmm_result.csv'
        df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))

        print(mannul_meaured_times)
        # milisecond
        return mannul_meaured_times[0]
