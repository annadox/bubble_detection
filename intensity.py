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

MERITEV = "Boiling meritve Vertical"

DATASET_DIR = os.path.join("D:/Mehurcki/data", MERITEV)
RECORDING_DIR = "Recording_Date=240321_Time=143739_50C_450kgm2s_p1" # Directory of the recording
IMAGES_DIR =  os.path.join(DATASET_DIR, RECORDING_DIR)

PIXEL_FILE = os.path.join("pixel_data", MERITEV, RECORDING_DIR + ".txt")

IMAGE_SIZE = (1280, 800) # Set this to the size of the RAWW input image (y, x)
IMAGE_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE # Set depending on how you want the input image to be rotated - set to None for no rotation

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

rr = RawwReader(IMAGE_SIZE, IMAGE_ROTATION)
rawws = sorted(Path(IMAGES_DIR).glob("*raww"))[0:NUMBER_IMAGES]
rawws_img = list(map(rr.read_raw_image, rawws))

pixels = read_pixels(PIXEL_FILE)

print(pixels)
px_check = range(-5, 6)

T = 1/400
x = np.linspace(0.0, 10.0 * NUMBER_IMAGES/4000, NUMBER_IMAGES)



""" for img in rawws_img:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for px in pixels:
        img = cv2.circle(img, (px[1], px[0]), radius=2, color=(0,255,0), thickness=-1)
    
    cv2.imshow("i", img)
    cv2.waitKey()  """
   



for px in pixels[:]:
    avg_t = []
    for img in rawws_img:
        avg = 0
        for diff in px_check:
            avg += img[px[0] + diff, px[1] ]/len(px_check)

        avg_t.append(avg)
    
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
    