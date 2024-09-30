import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog

# This script converts .xml data in "CVAT for images 1.1" format to .csv data, which is used for training.
# It bulk converts a whole dataset of labels, since we usually want to convert the whole dataset at once

# DATASET_DIR is the root directory of the dataset you're converting. It needs to contain subdirectories "xml_cvat" (input data) and "csv_data" (output data).
# If DATASET_DIR is None, the program will prompt you to select the dataset directory

DATASET_DIR = "./datasets/vertical_50_150_p3"

# Read the .xml file
def read_CVAT(path):
    ['centerX', 'centerY', 'orientation', 'major_axis_length', "minor_axis_length"]
    bubs = []
    tree = ET.parse(path)
    root = tree.getroot()
    for bubble in root.iter("ellipse"):
        rot = 0
        if("rotation" in bubble.attrib.keys()):
            
            rot = np.radians(float(bubble.attrib["rotation"]))

        major = max(bubble.attrib["rx"], bubble.attrib["ry"])
        minor = min(bubble.attrib["rx"], bubble.attrib["ry"])
        bubs.append([bubble.attrib["cy"], bubble.attrib["cx"], rot, major, minor])

    return bubs

# Write files to a .csv file
def write_to_csv(labels, name, dir):
    df = pd.DataFrame(labels, columns=['centerY', 'centerX', 'orientation', 'major_axis_length', "minor_axis_length"])
    df.to_csv(os.path.join(dir, name + ".csv"), index=False)

# Convert from .xml to .csv
def training_CVAT_to_csv(dataset_dir):
    dataset_xml_dir = os.path.join(dataset_dir, "xml_cvat")
    dataset_csv_dir = os.path.join(dataset_dir, "csv_data")
    for file in os.listdir(dataset_xml_dir):
        lab = read_CVAT(os.path.join(dataset_xml_dir, file))
        write_to_csv(lab, file.rsplit(".")[0], dataset_csv_dir)

if __name__ == "__main__":

    if DATASET_DIR == None:
        root = tk.Tk()
        root.withdraw()
        dataset_folder = filedialog.askdirectory()
    else:
        dataset_folder = DATASET_DIR

    training_CVAT_to_csv(dataset_folder)