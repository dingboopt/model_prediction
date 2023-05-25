import op
import argparse
import torch

def eva_op(d, op_type, args):
    if op_type == 'flash_attention':
        perf = op.PerfFlashAttention(args[0], args[1], args[2], args[3])
    with torch.cuda.device(d):
        perf.warmup()
        eva_time = perf.measure_time()

    return eva_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--dev', type=int, help='the tensor split size')
    parser.add_argument('--op_type', type=str, help='the op type to perf')
    parser.add_argument('--args', nargs='+', type=int, help='the op type to perf')

    args = parser.parse_args()

    
    print(f'op time is {eva_op(args.dev, args.op_type, args.args)}')
