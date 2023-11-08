import os
import cv2
import numpy as np
import argparse
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

exr = cv2.imread(args.input, -1)

cv2.imshow(args.input, exr)
cv2.waitKey(0)