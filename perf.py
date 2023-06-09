import os
import sys
from time import sleep
import datetime
from pytz import timezone
from utils import get_statistics
zone = 'Asia/Shanghai'
now_time = datetime.datetime.now(timezone(zone))
import torch
import pandas as pd


### bloom series model #####
### 176B: 

def generate_exp_series(end, start=0, base=2, reverse = True):
    result = []
    for i in range(start, end):
        result.append(base**i)
    if reverse:
        result.reverse()
    return result

mbs = generate_exp_series(5)
sequence = generate_exp_series(13, start=2)
tp = generate_exp_series(8, reverse = False)
# bloom 175b/7b/560m hyperparams
hidden = [14336, 4096, 1024]
atthead = [112, 32, 16]

for index in range(3):
    h = hidden[index]
    a = atthead[index]
    for b in mbs:
        for s in sequence:
            for t in tp:
                #for h in hidden:
                #    for a in atthead:
                        # --dev 1 --tp 8  --mbs 4 --hidden 14336 --sequence 4096 --atthead 112 --csv_filename test
                if a % t != 0:
                    continue
                try:
                    result = get_statistics(int(sys.argv[1]), b, h, s, a, t)
                except torch.cuda.OutOfMemoryError:
                    print(f'OOM for {[b,h,a,s,t]}')
                    continue
                if sys.argv[2] is None:
                    sys.exit(0)
                
                df = pd.DataFrame([result])

                output_path= sys.argv[2]
                df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
#                cmd = ' '.join(['python', 
#                        'main.py',
#                        f'--dev {sys.argv[1]}', 
#                        f'--mbs {b}', 
#                        f'--hidden {h}', 
#                        f'--sequence {s}',
#                        f'--atthead {a}',
#                        f'--tp {t}',
#                        f'--csv_filename {sys.argv[2]}'])
#                print(cmd)
#                #sleep(10)
#                os.system(cmd)
