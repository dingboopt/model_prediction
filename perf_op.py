import op
import argparse
from utils import eva_op
from utils import generate_exp_series
from utils import eva_naive_attention
from utils import eva_flash_attention
from utils import cal_flops
import torch
import pandas as pd
import os



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--dev', type=int, help='the tensor split size')
    parser.add_argument('--tp', type=int, help='the tensor split size')
    parser.add_argument('--mbs', type=int, help='the micro batch size')
    parser.add_argument('--hidden', type=int, help='the hidden size')
    parser.add_argument('--sequence', type=int, help='the sequence size')
    parser.add_argument('--atthead', type=int, help='the number of attenstion heads')
    parser.add_argument('--layer', type=int, help='the number of layers')
    parser.add_argument('--csv_filename', type=str, help='the cvs file name to store result')
    args = parser.parse_args()

    # we evaluate different self attention implementation
    # a. megatron naive
    # b. flash attention

    mbs = generate_exp_series(5)
    sequence = generate_exp_series(13, start=2)
    tp = generate_exp_series(8, reverse = False)
    # bloom 175b/7b/560m hyperparams
    hidden = [14336, 4096, 1024]
    atthead = [112, 32, 16]

    df = pd.DataFrame(columns=['b', 'h', 'a', 's', 't', 'time:naive', 'time:flash', 'sol:naive', 'sol:flash'])
    INF = 10**6
    d = args.dev
    for index in range(3):
        h = hidden[index]
        a = atthead[index]
        for b in mbs:
            for s in sequence:
                for t in tp:
                    if a % t != 0:
                        continue
                    flops = cal_flops(b, h, s, t)
                    naive_time = 0
                    # first try naive attention
                    try:
                        naive_time= eva_naive_attention(d, b, h, a, s, t)
                    except torch.cuda.OutOfMemoryError:
                        naive_time = INF
                        print('hello world')
                    naive_sol = flops * 10**3/naive_time /10**12/312

                    flash_time = eva_flash_attention(d, b, h, a, s, t) 
                    flash_sol = flops * 10**3/flash_time/10**12/312
                    result = [b,h,a,s,t,naive_time,flash_time,naive_sol,flash_sol]
                    df.loc[len(df.index)] = result
                    print(df.tail(5))
    
    



    output_path=f'{args.csv_filename}'
    df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    
