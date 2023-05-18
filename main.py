import op
import argparse

def evaMLP(d, b, h, s, t):
    # first projection
    s -= 1

    m = b * s
    n = h * 4 // t
    k = h

    eva_time = op.perf_linear(d, m, n, k)
    print('###########################')
    # gelu
    eva_time += op.perf_gelu(d, (s,b,n),(n))
    print('###########################')
    # second projection
    tmp = n
    n = k
    k = tmp
    eva_time += op.perf_linear(d, m, k ,n)

    return eva_time

def evaAttention(d, b, h, ah, s, t):
    s -= 1
    # np:num_attention_heads_per_partition
    np = ah // t
    # hn:hidden_size_per_attention_head
    hn = h // ah

    # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
    m = s * b
    k = h
    n = np * 3 * hn
    eva_time = op.perf_linear(d, m, n, k)
    # raw score: [b * np, sq, hn] [b * np, hn, sk]
    batch = b * np
    m = s
    k = hn
    n = s
    eva_time += op.perf_baddbmm(d, batch, m, k, n)


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


    print(evaAttention(args.dev, args.mbs, args.hidden, args.atthead, args.sequence, args.tp))
    print(evaMLP(args.dev, args.mbs, args.hidden, args.sequence, args.tp))
