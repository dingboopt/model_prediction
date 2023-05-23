import op
import argparse
import torch

def evaMLP(d, b, h, s, t):
    # first projection
    s -= 1

    m = b * s
    n = h * 4 // t
    k = h
    eva_time = 0
    perf_linear = op.PerfLinear(m, k, n)
    with torch.cuda.device(d):
        perf_linear.warmup()
        eva_time += perf_linear.measure_time()
    # gelu
    eva_time += op.perf_gelu(d, (s,b,n),(n))
    # second projection
    tmp = n
    n = k
    k = tmp
    perf_linear = op.PerfLinear(m, k, n)
    with torch.cuda.device(d):
        perf_linear.warmup()
        eva_time += perf_linear.measure_time()

    return eva_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--op_type', type=str, help='the op type to perf')
    parser.add_argument('--args', nargs='+', type=int, help='the op type to perf')

    args = parser.parse_args()

    print(f'{args.args}')





