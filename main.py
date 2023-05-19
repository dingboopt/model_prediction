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

def evaAttention(d, b, h, ah, s, t):
    eva_time = 0
    s -= 1
    # np:num_attention_heads_per_partition
    np = ah // t
    # hn:hidden_size_per_attention_head
    hn = h // ah

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    m = s * b
    k = h
    n = np * 3 * hn

    perf_linear = op.PerfLinear(m, k, n)
    with torch.cuda.device(d):
        perf_linear.warmup()
        eva_time += perf_linear.measure_time()

    # attention scores and attention mask [b, np, sq, sk]
    # raw score: [b * np, sq, hn] [b * np, hn, sk]
    batch = b * np
    m = s
    k = hn
    n = s

    perf_baddbmm = op.PerfBaddBmm(batch, m, k, n)
    with torch.cuda.device(d):
        perf_baddbmm.warmup()
        eva_time += perf_baddbmm.measure_time()

    perf_softmax = op.PerfFusedScaleMaskSoftmax(b, np, s)
    with torch.cuda.device(d):
        perf_softmax.warmup()
        eva_time += perf_softmax.measure_time( )

    # dropout [b, np, sq, sk]
    perf = op.PerfDropout((b, np, s, s))

    with torch.cuda.device(d):
        perf.warmup()
        eva_time += perf.measure_time()


    # context(kq * v) [b*np, s, s] [b*np, s, hn]
    perf = op.PerfBmm(batch, s, s , hn)
    with torch.cuda.device(d):
        perf.warmup()
        eva_time += perf.measure_time()

    perf = op.PerfLinear(b * s, hn * np, h)

    with torch.cuda.device(d):
        perf.warmup()
        eva_time += perf.measure_time()

    return eva_time


def evaInputLayerNorm(d, b, h, s):
    perf = op.PerfFusedLayerNorm(b, h, s)

    with torch.cuda.device(d):
        perf.warmup()
        eva_time = perf.measure_time()

    return eva_time

def evaBiasDropoutAddFused(d, b, h, s):
    perf = op.PerfBiasDropoutAddFused(b, h, s)

    with torch.cuda.device(d):
        perf.warmup()
        eva_time = perf.measure_time()

    return eva_time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--dev', type=int, help='the tensor split size')
    parser.add_argument('--tp', type=int, help='the tensor split size')
    parser.add_argument('--mbs', type=int, help='the micro batch size')
    parser.add_argument('--hidden', type=int, help='the hidden size')
    parser.add_argument('--sequence', type=int, help='the sequence size')
    parser.add_argument('--atthead', type=int, help='the number of attenstion heads')
    parser.add_argument('--layer', type=int, help='the number of layers')
    args = parser.parse_args()

    # checkpointing decoder layer: input layernorm + attention + residual(bias add dropout residual) + layernorm + mlp + add&norm
    checkpoint_layer_time = 0
    eva_time = evaInputLayerNorm(args.dev, args.mbs, args.hidden, args.sequence)
    print(f'evaInputLayerNorm is {eva_time} ms')
    checkpoint_layer_time += eva_time * 3

    eva_time = evaAttention(args.dev, args.mbs, args.hidden, args.atthead, args.sequence, args.tp)
    checkpoint_layer_time += eva_time
    print(f'evaAttention is {eva_time} ms')

    eva_time = evaBiasDropoutAddFused(args.dev, args.mbs, args.hidden, args.sequence)
    checkpoint_layer_time += eva_time*2
    print(f'evaBiasDropoutAddFused is {eva_time} ms')

    eva_time = evaMLP(args.dev, args.mbs, args.hidden, args.sequence, args.tp)
    print(f'evaMLP is {eva_time} ms')
    checkpoint_layer_time += eva_time

    print(f'total checkpointing forward time is {checkpoint_layer_time}')



