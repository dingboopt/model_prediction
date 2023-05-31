import os
import op
import torch

def eva_op(d, op_type, args):
    if op_type == 'flash_attention':
        perf = op.PerfFlashAttention(args[0], args[1], args[2], args[3])
    with torch.cuda.device(d):
        perf.warmup()
        eva_time = perf.measure_time()

    return eva_time

def cal_flops(b, h, s, t):
    # only cal matrix op
    flops = 0
    # qk
    flops += b*s*h*s * 2
    # qkv
    flops += b*s*h*s * 2
    
    return flops/t

def eva_flash_attention(d, b, h, ah, s, t):
    assert ah%t == 0 and h%ah ==0
    return eva_op(d, 'flash_attention', [b, ah//t, s, h//ah])

def eva_naive_attention(d, b, h, ah, s, t):
    detail_time = {}
    eva_time = 0
    s -= 1
    # np:num_attention_heads_per_partition
    np = ah // t
    # hn:hidden_size_per_attention_head
    hn = h // ah

    # attention scores and attention mask [b, np, sq, sk]
    # raw score: [b * np, sq, hn] [b * np, hn, sk]
    batch = b * np
    m = s
    k = hn
    n = s

    perf_baddbmm = op.PerfBaddBmm(batch, m, k, n)
    with torch.cuda.device(d):
        perf_baddbmm.warmup()
        tmp = perf_baddbmm.measure_time()
        detail_time['raw_attention'] = tmp
        eva_time += tmp
		
        print(f'raw attention time is {tmp}')

    perf_softmax = op.PerfFusedScaleMaskSoftmax(b, np, s)
    with torch.cuda.device(d):
        perf_softmax.warmup()
        tmp = perf_softmax.measure_time()
        detail_time['softmax'] = tmp
        print(f'softmax_time time is {tmp}')
        eva_time += tmp

    # dropout [b, np, sq, sk]
    perf = op.PerfDropout((b, np, s, s))

    with torch.cuda.device(d):
        perf.warmup()
        tmp = perf.measure_time()
        eva_time += tmp
        detail_time['dropout'] = tmp
        print(f'dropout_time time is {tmp}')


    # context(kq * v) [b*np, s, s] [b*np, s, hn]
    perf = op.PerfBmm(batch, s, s , hn)
    with torch.cuda.device(d):
        perf.warmup()
        tmp = perf.measure_time()
        detail_time['context_value'] = tmp
        eva_time += tmp
        print(f'context_value_time is {tmp}')

    return eva_time

def generate_exp_series(end, start=0, base=2, reverse = True):
    result = []
    for i in range(start, end):
        result.append(base**i)
    if reverse:
        result.reverse()
    return result

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
        print(f'hto4h time is {eva_time}')
    # gelu
    #eva_time += op.perf_gelu(d, (s,b,n),(n))
    # second projection
    tmp = n
    n = k
    k = tmp
    perf_linear = op.PerfLinear(m, k, n)
    with torch.cuda.device(d):
        perf_linear.warmup()
        fourh2h_time = perf_linear.measure_time()
        eva_time += fourh2h_time
        print(f'4htoh time is {fourh2h_time}')

    return eva_time

def evaAttention(d, b, h, ah, s, t):
    detail_time = {}
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
        tmp = perf_linear.measure_time()
        detail_time['QKV'] = tmp
        eva_time += tmp
        print(f'eval qkv time is {eva_time}')

    # attention scores and attention mask [b, np, sq, sk]
    # raw score: [b * np, sq, hn] [b * np, hn, sk]
    batch = b * np
    m = s
    k = hn
    n = s

    perf_baddbmm = op.PerfBaddBmm(batch, m, k, n)
    with torch.cuda.device(d):
        perf_baddbmm.warmup()
        tmp = perf_baddbmm.measure_time()
        detail_time['raw_attention'] = tmp
        eva_time += tmp
		
        print(f'raw attention time is {tmp}')

    perf_softmax = op.PerfFusedScaleMaskSoftmax(b, np, s)
    with torch.cuda.device(d):
        perf_softmax.warmup()
        tmp = perf_softmax.measure_time()
        detail_time['softmax'] = tmp
        print(f'softmax_time time is {tmp}')
        eva_time += tmp

    # dropout [b, np, sq, sk]
    perf = op.PerfDropout((b, np, s, s))

    with torch.cuda.device(d):
        perf.warmup()
        tmp = perf.measure_time()
        eva_time += tmp
        detail_time['dropout'] = tmp
        print(f'dropout_time time is {tmp}')


    # context(kq * v) [b*np, s, s] [b*np, s, hn]
    perf = op.PerfBmm(batch, s, s , hn)
    with torch.cuda.device(d):
        perf.warmup()
        tmp = perf.measure_time()
        detail_time['context_value'] = tmp
        eva_time += tmp
        print(f'context_value_time is {tmp}')

    perf = op.PerfLinear(b * s, hn * np, h)

    with torch.cuda.device(d):
        perf.warmup()
        tmp = perf.measure_time()
        eva_time += tmp
        detail_time['projection'] = tmp
        print(f'project_time is {tmp}')

    return eva_time, detail_time


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

def cal_flops(b, h, ah, s, t):
    flops = {}
    flops['flops:attention:QKV'] = b * s * h / t * 3 * h * 2
    flops['flops:attention:QK'] = b * ah / t * s * s * h / ah * 2
    flops['flops:attention:value'] = b * ah / t * s * s * h / ah * 2
    flops['flops:attention:attention'] = flops['flops:attention:QK'] + flops['flops:attention:value']
    flops['flops:attention:projection'] = b * s * h * h / t * 2
    flops['flops:mlp:h24h'] = b * s * h * 4 * h / t  * 2
    flops['flops:mlp:4h2h'] = b * s * h * 4 * h / t * 2
    flops['flops:mlp'] = flops['flops:mlp:h24h'] + flops['flops:mlp:4h2h']
    flops['flops:total'] = flops['flops:attention:QKV'] + flops['flops:attention:attention'] +flops['flops:attention:projection'] + flops['flops:mlp']
    return flops

def cal_sol(flops, time):
    sol = {}
    sol['sol:attention:QK'] = flops['flops:attention:QK'] / time['attention:QK'] * 10**3 / 10**12 / 312
    sol['sol:attention:QKV'] = flops['flops:attention:QKV'] / time['attention:QKV'] * 10**3 / 10**12 / 312
    sol['sol:attention:attention'] = flops['flops:attention:attention'] / time['attention:attention'] * 10**3 / 10**12 / 312
    sol['sol:attention:projection'] = flops['flops:attention:projection'] / time['attention:projection'] * 10**3 / 10**12 / 312
    sol['sol:mlp'] = flops['flops:mlp']/time['mlp'] * 10**3 / 10**12 / 312
    sol['sol:total'] = flops['flops:total']/time['total_time'] * 10**3 / 10**12 / 312
    return sol

def cal_percentage(time):
    result = {}
    result['per:attention:QKV'] = time['attention:QKV'] / time ['total_time']
    result['per:attention:softmax_dropout'] = (time['attention:softmax']+ time['attention:dropout'])/ time ['total_time']
    result['per:attention:attention'] = time['attention:attention'] / time ['total_time']
    result['per:attention:projection'] = time['attention:projection'] / time ['total_time']
    result['per:mlp'] = time['mlp'] / time ['total_time']
    return result

def get_statistics(d, b, h, s, ah, t):
    hyperparams = {}
    hyperparams['mbs'] = b
    hyperparams['hidden size'] = h
    hyperparams['sequence length'] = s
    hyperparams['num of heads'] = ah
    hyperparams['num of tp'] = t

    measured_times = {}
    # checkpointing decoder layer: input layernorm + attention + residual(bias add dropout residual) + layernorm + mlp + add
    checkpoint_layer_time = 0
    eva_time = evaInputLayerNorm(d, b, h, s)
    print(f'evaInputLayerNorm is {eva_time} ms')
    checkpoint_layer_time += eva_time * 2

    eva_time, detail_time = evaAttention(d, b, h, ah, s, t)
    checkpoint_layer_time += eva_time
    measured_times['attention'] = eva_time
    measured_times['attention:QKV'] = detail_time['QKV']
    measured_times['attention:QK'] = detail_time['raw_attention']
    measured_times['attention:softmax'] = detail_time['softmax']
    measured_times['attention:dropout'] = detail_time['dropout']
    measured_times['attention:value'] = detail_time['context_value']
    measured_times['attention:projection'] = detail_time['projection']
    # only attention{QKV--->context value}
    measured_times['attention:attention'] = measured_times['attention:QK'] + measured_times['attention:softmax'] + measured_times['attention:dropout'] + measured_times['attention:value']
    
    print(f'evaAttention is {eva_time} ms, detial time is {detail_time}')

    eva_time = evaBiasDropoutAddFused(d, b, h, s)
    checkpoint_layer_time += eva_time*2
    print(f'evaBiasDropoutAddFused is {eva_time} ms')

    eva_time = evaMLP(d, b, h, s, t)
    checkpoint_layer_time += eva_time
    measured_times['mlp'] = eva_time
    print(f'evaMLP is {eva_time} ms')
    print(f'total checkpointing forward time is {checkpoint_layer_time}')
    # logging into csv file
    measured_times['total_time'] = checkpoint_layer_time
    ### FLOPS and SOL (312 tflops tensor core)
    flops = cal_flops(b, h, ah, s, t)
    sol = cal_sol(flops, measured_times)
    time_percentage = cal_percentage(measured_times)

    result = {}
    #result |= flops | sol
    result = hyperparams |time_percentage| sol
    #result = hyperparams | sol
    print(f'result is {result}')
    checkpoint_layer_time += eva_time

    return result
