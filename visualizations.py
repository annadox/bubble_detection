import pandas as pd
import os
import sys
import numpy as np
import math
import matplotlib.pyplot as plt
from bisect import bisect
import tkinter as tk
from tkinter import filedialog

# This script is used to visualize the distributions in a set of measurements
# The script will iterate over the directory in steps of 4 folders, so make sure that for each measurement conditions there are 4 folders (camera positions)
# For each measurement conditions we display 4 graphs (p1, p2, p3 and p4)

# Set MEASUREMENTS_DIR to the desired directory that contains a set of measurements, if None the program will prompt you to manually select the directory
MEASUREMENTS_DIR = None

# Make histrogram for a single measurement
def make_bar(ax, data, bins, name, metric, n, range):
    ax.set_xlim([0, range])
    ax.set_ylim([0, 0.5])
    ax.bar(x=bins[:-1], align="edge", height=data, width=np.diff(bins), color="skyblue", edgecolor="black")
    ax.set_xlabel(metric)

    ax.set_ylabel('Volume percentage (%)')
    ax.set_title(name + " (n=" + str(n) + ")")

if __name__ == "__main__":
    if MEASUREMENTS_DIR == None:
        root = tk.Tk()
        root.withdraw()
        measurement_set_dir = filedialog.askdirectory()
    else:
        measurement_set_dir = MEASUREMENTS_DIR

    graphs_dir = os.path.join(measurement_set_dir, "graphs")
    predictions_dir = os.path.join(measurement_set_dir ,"csv_data")
    measurement_set_name = os.path.basename(measurement_set_dir)

    # Make graphs directory if it doesnt exist
    if not os.path.exists(graphs_dir):
        os.mkdir(graphs_dir)

    # Iterate over the csv_data directory in the measurement_set_dir to display bubble distributions for each set of measurement conditions(steps of 4)
    for i in range(0, len(os.listdir(predictions_dir)), 4): 
        fig, axes = plt.subplots(1, 4, figsize=(18,4))

        dirs = os.listdir(predictions_dir)[i:i+4]
        filename = measurement_set_name + "_" + dirs[0].rsplit("_")[-3] + "_" + dirs[0].rsplit("_")[-2]
        fig.suptitle(measurement_set_name + " measurements at: " + dirs[0].rsplit("_")[-3] + " and " + dirs[0].rsplit("_")[-2])
        axes = axes.ravel()
        for num_dir, dir in enumerate(dirs): 
            
            print(num_dir + 1, "-", dir)
            measurement_dir = os.path.join(predictions_dir, dir)
            all_bubbles_radius = []
            # Add equivalent radiuses of all the bubbles in the current measurement directory to an array
            for file in os.listdir(measurement_dir):
                df = pd.read_csv(os.path.join(measurement_dir, file))
                df["equivalent_radius"] = np.sqrt(df["major_axis_length"] * df["minor_axis_length"])
                df["eccentricity"] = np.sqrt(1 - np.square(df["minor_axis_length"])/np.square(df["major_axis_length"]))

                eq_r = np.asarray(df["equivalent_radius"])
                ecc = np.asarray(df["eccentricity"])

                all_bubbles_radius.extend(eq_r)

            # Get bins and values for the distributions
            num_bins = 60
            hist, bins = np.histogram(all_bubbles_radius, bins=num_bins, range=(0, num_bins))
            bins[-1] += 0.1
            sums = np.zeros(num_bins)
            for i in all_bubbles_radius:
                sums[bisect(bins, i) - 1] += np.power(i, 3)
            
            sums = sums/sum(sums)

            # Make histrogram
            make_bar(ax = axes[num_dir], data=sums, bins=bins, name = str(num_dir), metric = "Equivalent radius (px)", n=len(all_bubbles_radius), range=num_bins)
        
        fig.tight_layout()
        # Save figures before showing
        plt.savefig(os.path.join(graphs_dir, filename + ".jpg"))
    plt.show()    
        