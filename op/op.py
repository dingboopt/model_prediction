from abc import ABC, abstractmethod
import torch
import datetime
from pytz import timezone
import statistics


class PerfOP(ABC):
    def __init__(self):
        self.warmup_times = 10
        self.measure_times= 10

    @abstractmethod
    def prepare_data(self):
        pass
    @abstractmethod
    def run_kernel(self):
        pass

    def warmup(self):
        self.prepare_data()
        for i in range(self.warmup_times):
            self.run_kernel()

    def measure_time(self, with_profile = False, log_dir=None):
        self.prepare_data()
        if not log_dir:
            now_time = datetime.datetime.now(timezone('Asia/Shanghai'))
            log_dir = f'./log/{self.__class__.__name__}{now_time.strftime("%Y-%m-%d %H:%M:%S")}'


        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10))
            torch.profiler.tensorboard_trace_handler(log_dir)(prof)

        p = torch.profiler.profile(
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
            )

        if with_profile:
            p.start()

        steps = self.measure_times
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        start_times = [0 for _ in range(steps)]
        end_times = [0 for _ in range(steps)]

        torch.cuda.synchronize()
        start = datetime.datetime.now()

        for i in range(steps):
            torch.cuda.synchronize()
            start_events[i].record()
            start_times[i] = datetime.datetime.now()
            self.run_kernel()
            torch.cuda.synchronize()
            end_times[i] = datetime.datetime.now()
            end_events[i].record()
            if with_profile:
                p.step()
        if with_profile:
            p.stop()
        torch.cuda.synchronize()
        end = datetime.datetime.now()
        time_interval  = (end-start).total_seconds() * 1000
        time_interval /= steps

        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_mean = statistics.mean(times)
        mannul_meaured_times = [(end-start).total_seconds() * 1000 for start,end in zip(start_times, end_times)]
        mannul_meaured_times_mean = statistics.mean(mannul_meaured_times)
        print(mannul_meaured_times)

        return mannul_meaured_times_mean
