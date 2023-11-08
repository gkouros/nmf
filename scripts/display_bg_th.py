import os
import cv2
import torch
import numpy as np
import argparse
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

bg_state = torch.load(args.input)
bg = bg_state['bg_mat'].squeeze().permute(1,2,0).cpu().numpy()
bg8 = cv2.normalize(bg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
bg8 = cv2.cvtColor(bg8, cv2.COLOR_BGR2RGB)

cv2.imshow('Environment map', bg8)
cv2.waitKey(0)