import os
import sys
from time import sleep
import datetime
from pytz import timezone
zone = 'Asia/Shanghai'
now_time = datetime.datetime.now(timezone(zone))


### bloom series model #####
### 176B: 

def generate_exp_series(end, start=0, base=2, reverse = True):
    result = []
    for i in range(end-start+1):
        result.append(base**i)
    if reverse:
        result = reversed(result)
    return result

mbs = generate_exp_series(5)
sequence = generate_exp_series(12)
tp = generate_exp_series(8)
hidden = [14336, 4096, 1024]
atthead = [112, 32, 16]

for b in mbs:
    for s in sequence:
        for t in tp:
            for h in hidden:
                for a in atthead:
                    # --dev 1 --tp 8  --mbs 4 --hidden 14336 --sequence 4096 --atthead 112 --csv_filename test
                    cmd = ' '.join(['python', 
                            'main.py',
                            f'--dev {sys.argv[1]}', 
                            f'--mbs {b}', 
                            f'--hidden {h}', 
                            f'--sequence {s}',
                            f'--atthead {a}',
                            f'--tp {t}',
                            f'--csv_filename {sys.argv[2]}'])
                    print(cmd)
                    #sleep(10)
                    os.system(cmd)
