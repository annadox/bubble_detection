import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog

# This script converts .csv data to .txt data, which can be imported into our labeler to modify the labels for further training.
# It bulk converts a whole dataset of labels, its recommended you create a new dataset in the datasets folder and work with the dataset directory structure described in README.md
# If you're converting model predictions, copy the image and the .csv files into their coresponding folders in the root directory of the created dataset.

# DATASET_DIR is the root directory of the dataset you're converting. It needs to contain subdirectories "csv_data" (input data) and "text_data" (output data).
# If DATASET_DIR is None, the program will prompt you to select the dataset directory

DATASET_DIR = None

# Read .csv file
def read_csv(file_path):
    df = pd.read_csv(file_path)

    return df.to_numpy()

# Write to .txt file
def write_text(labels, name, dir):
    f = open(os.path.join(dir, name + ".txt"), "w+")
    for l in labels:
        s = "{0},{1},{2},{3},{4}\n".format(l[1], l[0], l[3], l[4], l[2])
        f.write(s)

# Convert .csv to .txt
def csv_to_text(dataset_dir):
    dataset_text_dir = os.path.join(dataset_dir, "text_data")
    dataset_csv_dir = os.path.join(dataset_dir, "csv_data")

    for file in os.listdir(dataset_csv_dir):
        lab = read_csv(os.path.join(dataset_csv_dir, file))
        write_text(lab, file.rsplit(".")[0], dataset_text_dir)

if __name__ == "__main__":
    if DATASET_DIR == None:
        root = tk.Tk()
        root.withdraw()
        dataset_folder = filedialog.askdirectory()
    else:
        dataset_folder = DATASET_DIR

    csv_to_text(dataset_folder)