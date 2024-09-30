import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog

# This script converts .txt data in our labelers format to .csv data, which is used for training.
# It bulk converts a whole dataset of labels, since we usually want to convert the whole dataset at once

# DATASET_DIR is the root directory of the dataset you're converting. It needs to contain subdirectories "txt_data" (input data) and "csv_data" (output data).
# If DATASET_DIR is None, the program will prompt you to select the dataset directory

DATASET_DIR = None

# Read from .txt file
def read_text(file_path):
    f = open(file_path, "r")
    
    bubbles_raw = f.readlines()
    bubbles_proc = []
    
    for bubble in bubbles_raw:
        temp = np.asarray(bubble.strip().split(","), dtype=float)
        bubbles_proc.append(temp)
        
    return bubbles_proc  

# Write to .csv file
def write_to_csv(labels, name, dir):
    ['centerX', 'centerY', 'orientation', 'major_axis_length', "minor_axis_length"]
    bubs = []
    for l in labels:
        if(len(l) == 3):
            bubs.append([l[1], l[0], 0,  l[2], l[2]])
        elif (len(l) == 5):
            bubs.append([l[1], l[0], l[4], max(l[2], l[3]), min(l[2], l[3])])
    
    df = pd.DataFrame(bubs, columns=['centerY', 'centerX', 'orientation', 'major_axis_length', "minor_axis_length"])
    df.to_csv(os.path.join(dir, name + ".csv"), index=False)

# Converts .txt to .csv
def text_to_csv(dataset_dir):
    dataset_text_dir = os.path.join(dataset_dir, "text_data")
    dataset_csv_dir = os.path.join(dataset_dir, "csv_data")

    for file in os.listdir(dataset_text_dir):
        lab = read_text(os.path.join(dataset_text_dir, file))
        write_to_csv(lab, file.rsplit(".")[0], dataset_csv_dir)

if __name__ == "__main__":
    if DATASET_DIR == None:
        root = tk.Tk()
        root.withdraw()
        dataset_folder = filedialog.askdirectory()
    else:
        dataset_folder = DATASET_DIR

    text_to_csv(dataset_folder)