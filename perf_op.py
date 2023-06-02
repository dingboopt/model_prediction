import op
import argparse
from utils import eva_op

import torch
import pandas as pd
import os


def exec_op(dev, op_type, args_list, args, output_path):
    if len(args_list) == 0:
        try:
            result = eva_op(dev, op_type, args)
        except torch.cuda.OutOfMemoryError:
            print(f'OOM or value error for {args}')
        #finally:
        #   print(f'hah for{args}')
            return   
        df = pd.DataFrame([result])
        print(df)
        if output_path:
            df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
        return
    else:
        for arg in args_list[0]:
            args.append(arg)
            exec_op(dev, op_type, args_list[1:], args, output_path)
            args.pop()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Cost Model for LLM(decoder only).')
    parser.add_argument('--dev', type=int, help='the tensor split size')
    parser.add_argument('--op_type', type=str,help='the op type need to perf')
    parser.add_argument('--op_args', type=int, nargs='+',help='the op type need to perf')
    parser.add_argument('--op_args_list', type=int, nargs='+', action='append',help='the op args')
    parser.add_argument('--csv_filename', type=str, help='the cvs file name to store result')
    args = parser.parse_args()

    op_type = args.op_type
    op_args_list = args.op_args_list


    arg_string = f'{op_type}_{op_args_list}'
    print(arg_string)
    output_path= arg_string.replace( ' ', '_').replace('[', 'I').replace(']', 'I').replace(':', '').replace(',', '')+args.csv_filename
    print(output_path)
    # we evaluate different self attention implementation
    # a. megatron naive
    # b. flash attention

    print(args.op_type, args.op_args, args.op_args_list)

    #eva_op(args.dev, args.op_type ,args.op_args)

    exec_op(args.dev, args.op_type, args.op_args_list, [], output_path)
    
