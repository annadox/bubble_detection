import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog

# This script converts .csv data to .xml data in "CVAT for images 1.1", which can be imported into CVAT to modify the labels for further training.
# It bulk converts a whole dataset of labels, its recommended you create a new dataset in the datasets folder and work with the dataset directory structure described in README.md
# If you're converting model predictions, copy the image and the .csv files into their coresponding folders in the root directory of the created dataset.

# DATASET_DIR is the root directory of the dataset you're converting. It needs to contain subdirectories "csv_data" (input data) and "xml_cvat" (output data).
# If DATASET_DIR is None, the program will prompt you to select the dataset directory

DATASET_DIR = None

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df.to_numpy()

def write_to_xml(labels, file_name, dir):
    f = open(os.path.join(dir, file_name + ".xml"), "w+")

    write_st = '<annotations> \n <image id="0" name="" width="1280" height="800"> \n' # Assumes image size 1280 x 800
    for l in labels:
        rot = np.degrees(l[2]) # CVAT needs degrees, rotations in the .csv files are in radians
        write_st += f'<ellipse label="bubble" source="manual" occluded="0" cx="{l[0]}" cy="{l[1]}" rx="{l[3]}" ry="{l[4]}" rotation="{rot}" z_order="0"> </ellipse> \n'
    write_st += "</image> \n </annotations>"

    f.write(write_st)

def csv_to_CVAT(dataset_dir):
    dataset_xml_dir = os.path.join(dataset_dir, "xml_cvat")
    dataset_csv_dir = os.path.join(dataset_dir, "csv_data")
    for file in os.listdir(dataset_csv_dir):
        lab = read_csv(os.path.join(dataset_csv_dir, file))
        write_to_xml(lab, file.rsplit(".")[0], dataset_xml_dir)

if __name__ == "__main__":

    if DATASET_DIR == None:
        root = tk.Tk()
        root.withdraw()
        dataset_folder = filedialog.askdirectory()
    else:
        dataset_folder = DATASET_DIR

    csv_to_CVAT(dataset_folder)