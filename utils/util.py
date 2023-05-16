import os
import subprocess
import torch
import numpy as np
import datetime
import statistics
import torch.nn.functional as F
import datetime
from pytz import timezone

import pandas as pd

import sys
import os


#DEVICE = 1
#CLOCK_SPEED = 1350 # Must choose a clock speed that's supported on your device.

def set_clock_speed(dev, speed):
    """
    Set GPU clock speed to a specific value.
    This doesn't guarantee a fixed value due to throttling, but can help reduce variance.
    """
    process = subprocess.Popen("nvidia-smi", stdout=subprocess.PIPE, shell=True)
    stdout, _ = process.communicate()
    process = subprocess.run(f"nvidia-smi -pm ENABLED -i {dev}",      shell=True)
    process = subprocess.run(f"nvidia-smi -lgc {speed}i -i {dev}", shell=True)

def reset_clock_speed(dev):
    """
    Reset GPU clock speed to default values.
    """
    subprocess.run(f"nvidia-smi -pm ENABLED -i {dev}", shell=True)
    subprocess.run(f"nvidia-smi -rgc -i {dev}", shell=True)

def warmup(dev):
    with torch.cuda.device(dev):
        cuda = torch.device(f'cuda:{dev}')
        xx = torch.rand((4096, 4096), device=cuda, dtype=torch.bfloat16)
        yy = torch.rand((4096, 4096), device=cuda, dtype=torch.bfloat16)
        zz = F.linear(xx, yy)
