import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import os
import tkinter as tk
from tkinter import filedialog

# This script is used to automatically all the needed sub-directories in a empty dataset directory

# The program will prompt you to select a directory in which you want to create a new dataset

SUB_DIRS = ["images", "text_data", "csv_data", "masks", "rawws", "xml_cvat"] # All the subdirs created

def create_sub(dataset_folder,  name):
    sub_dir = os.path.join(dataset_folder, name)
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)

def create_dataset(dataset_folder):
    for d in SUB_DIRS:
        create_sub(dataset_folder, d)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    dataset_folder = filedialog.askdirectory()

    if not os.listdir(dataset_folder):
        create_dataset(dataset_folder)
    else:
        raise Exception("Directory is not empty")
