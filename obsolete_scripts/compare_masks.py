from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
import tifffile
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes

import pandas as pd
import cv2
from utils import RawwReader
import random
import tkinter as tk
from tkinter import filedialog

# This script needs the predicted mask and the annotated mask of an image. It returns the average IoU of all bubble detections in the image.

# MASK_DETECTED is the path to the predicted mask of an image (.tif), MASK_TRUTH is the annotated mask of the image.
# If any of those two constants are None, the program will prompt you to manually select them. (first MASK_DETECTED, second MASk_TRUTH)

MASK_DETECTED = None
MASK_TRUTH = None

def read_mask(path):
    mask = tifffile.imread(path)
    return mask

def process(mask):
    return mask > 0

def get_intersection(m1, m2):
    it = np.zeros_like(m1)
    it = it > 0
    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            if(m1[i,j] and m2[i,j]):
                it[i,j] = True
    
    return it

def get_union(m1, m2):
    un = np.zeros_like(m1)
    un = un > 0
    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            if(m1[i,j] or m2[i,j]):
                un[i,j] = True

    return un

if __name__ == "__main__":

    if MASK_DETECTED == None:
        root = tk.Tk()
        root.withdraw()
        mask_detected = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
    else:
        mask_detected = MASK_DETECTED

    if MASK_TRUTH == None:
        root = tk.Tk()
        root.withdraw()
        mask_truth = filedialog.askopenfilename(filetypes=[("Image files", "*.tif")])
    else:
        mask_truth = MASK_TRUTH
    

    md = read_mask(mask_detected)
    mt = read_mask(mask_truth)

    md = process(md)
    mt = process(mt)

    print(mt)

    intersection = get_intersection(md, mt)
    union = get_union(md, mt)

    intersection_count = np.count_nonzero(intersection) / np.count_nonzero(union)

    print(intersection_count)
    



    
