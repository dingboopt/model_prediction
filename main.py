import op
import argparse
import torch
import pandas as pd
import os
import sys
from utils import get_statistics



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--dev', type=int, help='the tensor split size')
    parser.add_argument('--tp', type=int, help='the tensor split size')
    parser.add_argument('--mbs', type=int, help='the micro batch size')
    parser.add_argument('--hidden', type=int, help='the hidden size')
    parser.add_argument('--sequence', type=int, help='the sequence size')
    parser.add_argument('--header', type=int, help='the number of attenstion heads')
    parser.add_argument('--layer', type=int, help='the number of layers')
    parser.add_argument('--csv_filename', type=str, help='the cvs file name to store result')


    parser.add_argument('--tp_list', nargs="+",type=int, help='the tensor split size')
    parser.add_argument('--mbs_list', nargs="+",type=int, help='the micro batch size')
    parser.add_argument('--hidden_list',nargs="+", type=int, help='the hidden size')
    parser.add_argument('--sequence_list',nargs="+", type=int, help='the sequence size')
    parser.add_argument('--header_list',nargs="+", type=int, help='the number of attenstion heads')

    args = parser.parse_args()

    if args.tp is not None:
        tp = [args.tp]
    elif args.tp_list is not None:
        tp = args.tp_list
    
    if args.mbs is not None:
        mbs = [args.mbs]
    elif args.mbs_list is not None:
        mbs = args.mbs_list
 
    if args.hidden is not None:
        hidden = [args.hidden]
    elif args.hidden_list is not None:
        hidden = args.hidden_list
 
    if args.sequence is not None:
        sequence = [args.sequence]
    elif args.sequence_list is not None:
        sequence = args.sequence_list
        
    if args.header is not None:
        header = [args.header]
    elif args.header_list is not None:
        header = args.header_list

    
    arg_string = f'micro batch: {mbs} header:{header} sequence:{sequence} hidden size:{hidden} tp:{tp}'

    print(arg_string)
    output_path= arg_string.replace( ' ', '_').replace('[', '').replace(']', '').replace(':', '').replace(',', '')+args.csv_filename
    print(output_path)

    for h in hidden:
        for a in header:
            for s in sequence:
                for b in mbs:
                    for t in tp:
                        if a % t != 0:
                            print(f'warning : atten head num:{a} cannot be divided by tp num {t}')
                            continue
                        try:
                            result = get_statistics(args.dev, b, h, s, a, t)
                        except torch.cuda.OutOfMemoryError:
                            print(f'OOM for micro batch: {b} head:{a} sequence:{s} hidden size:{h} tp:{t}')
                            continue

                        df = pd.DataFrame([result])
                        print(df)
                        if output_path:
                            df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
