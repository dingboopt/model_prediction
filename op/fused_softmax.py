from .op import PerfOP
import torch


class PerfFusedScaleMaskSoftmax(PerfOP):
    def __init__(self, b, h, s):
        super(PerfFusedScaleMaskSoftmax, self).__init__()
        self.b = b
        self.h = h
        self.s = s
        self.input1 = None
        self.input2 = None

    def prepare_data(self):
        cuda = torch.cuda.current_device()
        self.input1 = torch.rand((self.b, self.h, self.s, self.s), device=cuda, dtype=torch.bfloat16)
        self.input2 = None
        self.mask = torch.rand((1, 1, self.s, self.s), device=cuda) < 0.9

    def run_kernel(self):

        # current setting bf16->fp32
        input = self.input1.float()
        # scale
        input = input * 5.6
        # fille mask
        input.masked_fill_(self.mask, torch.finfo(input.dtype).min)
        # use torch softmax to obtain prob
        probs = torch.nn.Softmax(dim=-1)(input)

        self.input1 = input.bfloat16()

# b: batch size h: heads s: sequence
def perf_fused_softmax(dev, b, h, s):
    with torch.cuda.device(dev):
        cuda = torch.device(f'cuda:{dev}')
        input1 = torch.rand((b, h, s, s), device=cuda, dtype=torch.bfloat16)
        input2 = None

        torch.cuda.synchronize()
        steps = 10
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(steps)]
        start_times = [0 for _ in range(steps)]
        end_times = [0 for _ in range(steps)]
        mask = torch.rand((1, 1, s, s), device=cuda) < 0.9

        def trace_handler(prof):
            pass
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=10))
            torch.profiler.tensorboard_trace_handler('./log/softmax')(prof)


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
                # current setting bf16->fp32
                input = input1.float()
                # scale
                input = input * 5.6
                # fille mask
                input.masked_fill_(mask, torch.finfo(input.dtype).min)
                # use torch softmax to obtain prob
                probs = torch.nn.Softmax(dim=-1)(input)

                input1 = input.bfloat16()

                torch.cuda.synchronize()
                end_times[i] = datetime.datetime.now()
                end_events[i].record()
                p.step()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        times_mean = statistics.mean(times)
        mannul_meaured_times = [(end-start).total_seconds() * 1000 for start,end in zip(start_times, end_times)]
        mannul_meaured_times_mean = statistics.mean(mannul_meaured_times)
        print(times)
        print('=====================================================')
        print(mannul_meaured_times)
        # milisecond
        return mannul_meaured_times_mean
