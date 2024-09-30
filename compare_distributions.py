import pandas as pd
import tkinter as tk
from tkinter import filedialog

from glob import glob
from csbdeep.utils import Path
import numpy as np
from bisect import bisect
import matplotlib.pyplot as plt
from scipy import stats

# This script is used to compare the distributions of bubble labels in two different datasets.

# The program will first prompt you to select the directory (dataset) with the manually annotated data, then the one with data predicted by a model. 
# The files in both directoreis need to be in a .csv file format, where each file represents the bubble labels of a single image.
# Files not in a .csv file format are automatically skipped.

# Make histrogram of bubble volume precentage for single directory
def make_bar(ax, data, bins, name, metric, n):
    ax.set_xlim([0, 60])
    ax.set_ylim([0, 0.5])
    ax.bar(x=bins[:-1], align="edge", height=data, width=np.diff(bins), color="skyblue", edgecolor="black")
    ax.set_xlabel(metric)

    ax.set_ylabel('Volume percentage (%)')
    ax.set_title(name + f" (n_bubbles={n})")

# Make line graph comparing distributions of both directories
def make_compare_line(ax, data, bins, name, metric, colors, labels):
    t_stat, p_value = stats.ttest_ind(data[0], data[1])

    ax.set_xlim([0, 60])
    ax.set_ylim([0, 0.5])
    ax.set_xlabel(metric)
    ax.set_ylabel('Volume percentage (%)')
    ax.set_title(name + f" (t_stat={t_stat}, p_value={p_value})")

    for (d, b, c, l) in zip(data, bins, colors, labels):
        ax.plot(b[:-1], d, label=l, linestyle='-', color=c)

    ax.legend()

# Make bins and values for the graph (current and recommended size of bin is 1px)
def make_distribution(csv_files, name):
    all_bubbles_radius = []
    for i, df in enumerate(csv_files):
        df["equivalent_radius"] = np.sqrt(df["major_axis_length"] * df["minor_axis_length"])
        df["eccentricity"] = np.sqrt(1 - np.square(df["minor_axis_length"])/np.square(df["major_axis_length"]))

        # Script is currently calculating the distributions based on equivalent radiu, but can be adapted to show distributions based on eccentricty
        eq_r = np.asarray(df["equivalent_radius"])
        ecc = np.asarray(df["eccentricity"])

        all_bubbles_radius.extend(eq_r)

    num_bins = 60
    hist, bins = np.histogram(all_bubbles_radius, bins=num_bins, range=(0, num_bins)) # Attribute range sets what range of bubble we are accepting.
    bins[-1] += 0.1
    sums = np.zeros(num_bins)
    for i in all_bubbles_radius:
        sums[bisect(bins, i) - 1] += np.power(i, 3)
    
    sums = sums/sum(sums)

    return sums, bins, all_bubbles_radius

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    folder_marked = filedialog.askdirectory()
    #folder_marked = r"D:/Mehurcki/BubDist/datasets/vertical_50_150_p3/csv_data"
    marked_csv = sorted(Path(folder_marked).glob("*csv"))
    marked_csv = list(map(pd.read_csv, marked_csv))
    print(folder_marked)

    folder_predicted = filedialog.askdirectory()
    #folder_predicted = r"D:/Mehurcki/BubDist/output/Boiling meritve vertical/csv_data/Recording_Date=240321_Time=111637_50C_150kgm2s_p3"
    predicted_csv = sorted(Path(folder_predicted).glob("*csv"))
    predicted_csv = list(map(pd.read_csv, predicted_csv))
    print(folder_predicted)
    
    sums_marked, bins_marked, abr_marked = make_distribution(marked_csv, "Distribution of annotated bubbles")
    sums_predicted, bins_predicted, abr_predicted = make_distribution(predicted_csv, "Distribution of predicted bubbles")

    labels_distributions = [f"Anottated distribution (n={len(abr_marked)})", f"Predicted distribution (n={len(abr_predicted)})"]
    colors_distributions = ["b", "r"]

    f = plt.figure()
    ax = f.add_subplot(111)
    make_compare_line(ax = ax, data=[sums_marked, sums_predicted], bins=[bins_marked, bins_predicted], colors=colors_distributions, labels=labels_distributions, name="Comparison betwee distributions", metric = "Equivalent radius (px)")

    # Make annotated
    f = plt.figure()
    ax = f.add_subplot(111)
    make_bar(ax = ax, data=sums_marked, bins=bins_marked, name = "Distribution of annotated bubbles", metric = "Equivalent radius (px)", n=len(abr_marked))
    
    # Graph predicted
    f = plt.figure()
    ax = f.add_subplot(111)
    make_bar(ax = ax, data=sums_predicted, bins=bins_predicted, name = "Distribution of predicted bubbles", metric = "Equivalent radius (px)", n=len(abr_predicted))

    print(marked_csv)

    plt.tight_layout()
    plt.show()


    
