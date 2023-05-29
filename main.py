import op
import argparse
import torch
import pandas as pd
import os
from utils import evaAttention
from utils import evaInputLayerNorm
from utils import evaBiasDropoutAddFused
from utils import evaMLP
from utils import cal_sol
from utils import cal_flops
from utils import cal_percentage
from utils import get_statistics



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

    get_statistics(args.dev, args.mbs, args.hidden, args.sequence, args.atthead, args.tp)

