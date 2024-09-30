import numpy as np
import pandas as pd
import os

DATASET_DIR = "./datasets/full_dataset"

OUTPUT_DIR = "./output"

DATASET_TEXT_DIR = os.path.join(DATASET_DIR, "text_data")
DATASET_CSV_DIR = os.path.join(DATASET_DIR, "csv_data")

OUTPUT_TEXT_DIR = os.path.join(OUTPUT_DIR, "text_data")
OUTPUT_CSV_DIR = os.path.join(OUTPUT_DIR, "csv_data")

def read_text(file_path):
    f = open(file_path, "r")
    
    bubbles_raw = f.readlines()
    bubbles_proc = []
    
    for bubble in bubbles_raw:
        temp = np.asarray(bubble.strip().split(","), dtype=float)
        bubbles_proc.append(temp)
        
    return bubbles_proc  

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

def read_csv(file_path):
    df = pd.read_csv(file_path)

    return df.to_numpy()

def write_text(labels, name, dir):
    f = open(os.path.join(dir, name + ".txt"), "w+")
    for l in labels:
        s = "{0},{1},{2},{3},{4}\n".format(l[1], l[0], l[3], l[4], l[2])
        f.write(s)

def training_text_to_csv():
    for file in os.listdir(DATASET_TEXT_DIR):
        lab = read_text(os.path.join(DATASET_TEXT_DIR, file))
        write_to_csv(lab, file.rsplit(".")[0], DATASET_CSV_DIR)

def output_csv_to_text():
    for dir in os.listdir(OUTPUT_CSV_DIR):
        curr_csv_dir = os.path.join(OUTPUT_CSV_DIR, dir)

        curr_text_dir = os.path.join(OUTPUT_TEXT_DIR, dir)
        if not os.path.exists(curr_text_dir):
            os.mkdir(curr_text_dir)

        for file in os.listdir(curr_csv_dir):
            lab = read_csv(os.path.join(curr_csv_dir, file))
            write_text(lab, file.rsplit(".")[0], curr_text_dir)

        print("Written to " + curr_text_dir)

if __name__ == "__main__":
    training_text_to_csv()
    #output_csv_to_text()
    

