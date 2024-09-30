import tkinter as tk
from tkinter import filedialog

from matplotlib import pyplot as plt
import numpy as np
import cv2
import os
import json
import skimage.morphology as morph
import pandas as pd
from scipy.ndimage import shift
from scipy import fft
from scipy.signal import find_peaks

from utils import RawwReader
from utils import draw_bubbles

from glob import glob
from csbdeep.utils import Path

import tifffile

NUMBER_IMAGES = 400 # Must be multiple of 400
MAX_IMAGES = 4000

def read_pixels(file_path):
    array_2d = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            numbers = line.split(',')
            row = [int(num) for num in numbers]
            array_2d.append(row)
    return array_2d

root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()

rawws = sorted(Path(folder_selected).glob("*tif"))
masks = list(map(tifffile.imread, rawws))

name = folder_selected.split("/")[-2].split("+")
pixels = read_pixels(os.path.join("pixel_data", name[0], name[1]) + ".txt")

px_check = range(-5, 6)

T = 1/400
x = np.linspace(0.0, 10.0 * NUMBER_IMAGES/4000, NUMBER_IMAGES)

""" for img in masks:
    plt.imshow(img)
    plt.waitforbuttonpress() """

for px in pixels[:]:
    avg_t = []
    for img in masks:
        a = False
        for diff in px_check:
            if img[px[0] + diff, px[1] ] != 0:
                a = img[px[0] + diff, px[1]]
                break
        if a != 0:
            avg_t.append(a)
        else:
            avg_t.append(0)

    avg_frequency_space = np.fft.fft(avg_t)

    # Compute the FFT
    avg_frequency_space = np.fft.fft(avg_t)
    xf = np.fft.fftfreq(len(avg_t), T)

    # Only keep the positive frequencies
    xf = xf[:len(x)//2]
    avg_frequency_space = avg_frequency_space[:len(x)//2]
    avg_frequency_space[0] = 0

    idx = np.argmax(np.abs(avg_frequency_space))
    peak_freq = xf[idx]
    peak_magnitude = np.abs(avg_frequency_space[idx])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    fig.suptitle('Intensity change around ({}, {}), freq={}'.format(px[0], px[1], peak_freq), fontsize=16)

    ax1.set_title('Average intensity')
    ax1.plot(x, avg_t) 

    ax2.set_title('FFT')
    
    ax2.plot(2.0/len(avg_t) * np.abs(avg_frequency_space))  
    fig.show()

plt.show()
    