import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
import os
import cv2
from utils import RawwReader
from utils import draw_bubbles
import tkinter as tk
from tkinter import filedialog

from PIL import Image

# Since we want all the .rawws rotated vertically with the coolant flowing upwards, you can use this script to rotate the .raww image if the camera was rotated incorectlly during measurements

# !! The original .rawws will be overwritten !!
# The script will prompt you to select a folder containing a set of measurements

IMG_SIZE = (800, 1280) # Set this to the size of the RAWW input image (height, width)W
POST_ROTATION = cv2.ROTATE_90_CLOCKWISE # Set this to the desired rotatiion of the image

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    dir_path = filedialog.askdirectory()

    for folder in os.listdir(dir_path):
        print(folder)
        rawws = sorted(Path(os.path.join(dir_path, folder)).glob("*raww"))
        i = 1
        for path in tqdm(rawws):
            # Reads raww image wihtout normalizing it
            raw_img = np.fromfile(path, dtype='int16', sep="")       
            raw_img = np.reshape(raw_img, IMG_SIZE)
            image = cv2.rotate(raw_img, cv2.ROTATE_90_CLOCKWISE)

            # Saves transformed image to the same file
            with open(os.path.join(dir_path, folder, "Measurement{:04d}.raww".format(i)), "wb") as file:
                file.write(image)

            i += 1

