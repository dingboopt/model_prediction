import op
import argparse

def evaMLP(d, b, h, s, t):
    # first projection
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
    eva_time += op.perf_linear(d, m, n ,k)

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



    print(evaMLP(args.dev, args.mbs, args.hidden, args.sequence, args.tp))