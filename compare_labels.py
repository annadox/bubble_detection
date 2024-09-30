from utils import RawwReader
from utils import draw_bubbles_color_df
import numpy as np
import pandas as pd
import cv2
import tkinter as tk
from tkinter import filedialog

# This script is used to visually compare the predicted and the annotated labels of an image.

# IMAGE_PATH is the path to the image. TRUTH_PATH is the path to the anottated labels in a .csv file format, PREDICTIONS_PATH is the path to the predicted labels in a .csv file format
# If any of those constants are None, the program will prompt you to manually select them. (first IMAGE_PATH, second TRUTH_PATH, third PREDICTIONS_PATH)

IMAGE_PATH = None

TRUTH_PATH = None
PREDICTIONS_PATH = None

if __name__ == "__main__":

    if IMAGE_PATH == None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    else:
        image_path = IMAGE_PATH

    if TRUTH_PATH == None:
        root = tk.Tk()
        root.withdraw()
        truth_path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv")])
    else:
        truth_path = TRUTH_PATH

    if PREDICTIONS_PATH == None:
        root = tk.Tk()
        root.withdraw()
        predictions_path = filedialog.askopenfilename(filetypes=[("Data files", "*.csv")])
    else:
        predictions_path = PREDICTIONS_PATH

    img = cv2.imread(image_path)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    truth = pd.read_csv(truth_path)
    preds = pd.read_csv(predictions_path)

    img = draw_bubbles_color_df(img, preds, clr=(0,255,0))
    img = draw_bubbles_color_df(img, truth, clr=(0,0,255))

    cv2.imshow("1", img)
    cv2.imwrite("D:/Mehurcki/BubDist/test2.png", img)
    cv2.waitKey()