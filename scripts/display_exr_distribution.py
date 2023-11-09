import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
from pathlib import Path
import seaborn as sns

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

exr = cv2.imread(args.input, -1)

# plt.hist(exr.clip(0, 1000).ravel(), bins=100)
# plt.yscale('log',base=2)
# plt.show()

sns.displot(exr[exr<10].ravel(), kde=True)
plt.yscale('log',base=2)
plt.show()