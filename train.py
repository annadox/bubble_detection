from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os
import numpy as np
import matplotlib
matplotlib.rcParams["image.interpolation"] = 'none'
import matplotlib.pyplot as plt
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

from glob import glob
from tqdm import tqdm
import tifffile
from csbdeep.utils import Path, normalize

from stardist import fill_label_holes, random_label_cmap, calculate_extents, gputools_available
from stardist.matching import matching, matching_dataset
from stardist.models import Config2D, StarDist2D, StarDistData2D

import pandas as pd
import cv2
from utils import RawwReader
import random

from dataset_scripts import cvat_to_csv, text_to_csv

# This script trains the model on a selected dataset

# Change the constants in this script to your needs before running it.
# The directory at path DATASET_DIR needs to be in the before mentioned dataset file structure (created by create_dataset.py script).

ROOT_DIR = os.getcwd() # Root directory of the script file
DATASET_DIR = os.path.join(ROOT_DIR, "datasets/full_dataset") # Relative path to the dataset from the root directory

MASKS_DIR = os.path.join(DATASET_DIR, "masks") # Relative path to masks folder in the dataset directory (.tif files)
CSV_DIR = os.path.join(DATASET_DIR, "csv_data") # Relative path to csv_data folder in the dataset directory (.csv files)
IMAGES_DIR = os.path.join(DATASET_DIR, "images")  # Relative path to images folder in the dataset directory (.png files)
RAWW_DIR = os.path.join(DATASET_DIR, "rawws") # Relative path to rawws folder in the dataset directory (.raww files)
XML_DIR = os.path.join(DATASET_DIR, "xml_cvat") # Relative path to the xml_cvat folder in the dataset directory (.xml files)
TEXT_DIR = os.path.join(DATASET_DIR, "text_data") # Relative path th the text_data folder in the dataset directory (.txt files)

# I recommend avoiding using .raww images in training and rather using already converted .png images, since the .raww images can sometimes be rotated which can ruin the training process.
# In .raww images from most measurements the rod is vertical and coolant is flowing upwards (height=1280, width=800).
# For training we need images where the rod is horizontal and the coolant is flowing to the right (height=800, width=1280). To get these image we rotate the .rawws by 90 degrees clockwise

# These two constants are only related to reading .raww images for the purpose of conversion to .png. They are not related in any ways to reading the .png images
IMAGE_SIZE = (1280, 800) # Set this to the size of the RAWW input image (height, width) - needs to be set to the dimensions of the original .raww image size (before rotation)
IMAGE_ROTATION = cv2.ROTATE_90_CLOCKWISE # Set depending on how you want the input image to be rotated - set to None for no rotation

PNG_IMAGE_SIZE = (800, 1280) # Set this to the .png image dimensions (height, width)

CHECK_IMAGES = True # Set to True if you want to display the dataset images and their corresponding labels before training
GENERATE_MASKS = True # Set to False if you already have generated masks

# Before training the model you need to set the models config.json file that determines the hyper-parameters of the training process (or you can just copy it from the full_model_best/stardist folder)
MODEL_BASE_DIR = "models/new_test" # Set to relative path to the models base directory from the root directory

# Mask generation from .csv files
def generate_masks(imgs):
    if(not os.path.exists(MASKS_DIR)):
        os.mkdir(MASKS_DIR)
    else:
        contents = sorted(Path(MASKS_DIR).glob("*tif"))
        for i in contents:
            os.remove(i)

    for file, img in zip(sorted(Path(CSV_DIR).glob("*csv")), imgs):
        file = file.name
        df = pd.read_csv(os.path.join(CSV_DIR, file))
        mask = np.zeros_like(img)
        for index, row in df.iterrows():
            mask = cv2.ellipse(mask, (int(row["centerX"]), int(row["centerY"])), (int(row["minor_axis_length"]), int(row["major_axis_length"])), row["orientation"], startAngle=0, endAngle=360, color=(index + 1), thickness=-1) 

        mask = mask.astype(np.uint16)
        tifffile.imwrite(os.path.join(MASKS_DIR, file.rsplit(".")[0] + ".tif"), mask)

# Train and validation dataset split
def train_val_split(images, masks, train_size):
    split = int(train_size * len(images))

    train_imgs = images[:-split]
    train_masks = masks[:-split]

    val_imgs = images[-split:]
    val_masks = masks[-split:]

    return train_imgs, train_masks, val_imgs, val_masks

# Preview images
def preview_images(images, masks):
    for img, mask in zip(images, masks):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(20,20))
        ax1.imshow(img,cmap='gray')
        ax2.imshow(img,cmap='gray');ax2.imshow(mask,cmap=random_label_cmap(),alpha=0.3)
        ax1.axis("off")
        ax2.axis("off")
        plt.tight_layout()
        plt.show()

# Function for changing intensity
def random_intensity_change(img):
    img = img*np.random.uniform(0.6,2) + np.random.uniform(-0.2,0.2)
    return img

# Function for flipping images
def random_fliprot(img, mask): 
    assert img.ndim >= mask.ndim
    axes = tuple(range(mask.ndim))
    perm = tuple(np.random.permutation(axes))
    img = img.transpose(perm + tuple(range(mask.ndim, img.ndim))) 
    mask = mask.transpose(perm) 
    for ax in axes: 
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=ax)
            mask = np.flip(mask, axis=ax)
    return img, mask 

# Commonly used augmenters on data (plug any combination of the three functions into the augmenter function to use them)

def augmenter_intensity(x, y):
    x = random_intensity_change(x)
    return x, y

def augmenter_noise(x, y):
    sig = 0.02*np.random.uniform(0,1)
    x = x + sig*np.random.normal(0,1,x.shape)
    return x, y

def augmenter_flip(x, y):
    x, y = random_fliprot(x,y)
    return x, y

def augmenter(x, y):
    """Augmentation of a single input/label image pair.
    x is an input image
    y is the corresponding ground-truth label image
    """
    return x, y

def train(images, masks):
    model = StarDist2D(None, name='stardist', basedir=MODEL_BASE_DIR) # Initialize stardist model

    # Get median object size and fov, print warning if median object size larger than fov
    median_size = calculate_extents(list(masks), np.median)
    fov = np.array(model._axes_tile_overlap('YX'))
    print(f"median object size:      {median_size}")
    print(f"network field of view :  {fov}")
    if any(median_size > fov):
        print("WARNING: median object size larger than field of view of the neural network.")

    train_imgs, train_masks, val_imgs, val_masks = train_val_split(images, masks, train_size=0.8) # Split dataset into training and validation - train_size determines the size of the training dataset compared to the whole dataset (recommended 0.7 or 0.8)
    try :
        # Train model and optimize the detection threshold to maximize the perfomance of the validation dataset
        model.train(train_imgs, train_masks, validation_data=(val_imgs, val_masks), augmenter=augmenter) # To add augmentations you can add code to the augmenter(x, y) function defined above.
    except KeyboardInterrupt:
        pass

    print("\nTraining stopped")

    print("Optimizing tresholds")
    model.optimize_thresholds(val_imgs, val_masks)
    # Predict masks for the validation dataset
    val_masks_pred = [model.predict_instances(x, n_tiles=model._guess_n_tiles(x), show_tile_progress=False)[0] for x in tqdm(val_imgs)]
    # Calculate performance metrics on specified IoU thresholds (precision, recall, accuracy, f1, mean_true_score, mean_matched_score, panoptic_quality, false positives, true positives and false negatives)
    print("Displaying statistics")
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # IoU thresholds
    stats = [matching_dataset(val_masks, val_masks_pred, thresh=t, show_progress=False) for t in tqdm(taus)] # Get performance metrics
    plot_stats(taus, stats)



# Plot performance metrics
def plot_stats(taus, stats):
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))

    for m in ('precision', 'recall', 'accuracy', 'f1', 'mean_true_score', 'mean_matched_score', 'panoptic_quality'):
        ax1.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax1.set_xlabel(r'IoU threshold $\tau$')
    ax1.set_ylabel('Metric value')
    ax1.grid()
    ax1.legend()

    for m in ('fp', 'tp', 'fn'):
        ax2.plot(taus, [s._asdict()[m] for s in stats], '.-', lw=2, label=m)
    ax2.set_xlabel(r'IoU threshold $\tau$')
    ax2.set_ylabel('Number #')
    ax2.grid()
    ax2.legend()
    
    plt.show()

if __name__ == "__main__":
    # Read .raww images from raww sub-directory, converts them and saves them to .png.
    rr = RawwReader(IMAGE_SIZE, IMAGE_ROTATION)
    rawws = sorted(Path(RAWW_DIR).glob("*raww"))
    rawws_img = list(map(rr.read_raw_image, rawws))

    for img, name in zip(rawws_img, rawws):
        n = os.path.basename(name).rsplit(".")[0] + ".png"
        print(n)
        cv2.waitKey()
        cv2.imwrite(os.path.join(IMAGES_DIR, n), img)

    # Read all .png images from the images sub-directory create a a full dataset of images and masks    
    image_paths = sorted(Path(IMAGES_DIR).glob("*png"))
    images = [cv2.imread(str(pt), cv2.IMREAD_GRAYSCALE) for pt in image_paths]
    images = [normalize(img, 0, 99.8, axis=(0,1), dtype=np.uint8) for img in images]

    """ for (img_path, img) in zip(image_paths, images):
        if img.shape != PNG_IMAGE_SIZE:
            raise Exception(f"Image at {img_path} with shape: {img.shape} does not match the required input shape {PNG_IMAGE_SIZE}") """

    # Convert .xml and .txt to .csv
    cvat_to_csv.training_CVAT_to_csv(DATASET_DIR)
    text_to_csv.text_to_csv(DATASET_DIR)

    # Program generates masks in .tif format for each image, if the masks sub-directory doesnt exist, it will create it. If it already contains files, it will clear the folder.
    if GENERATE_MASKS:
        generate_masks(images)

    # Read masks
    masks = sorted(Path(MASKS_DIR).glob("*tif"))
    masks = list(map(tifffile.imread, masks))
    masks = [fill_label_holes(mask) for mask in masks]

    # Shuffle the dataset
    c = list(zip(images, masks))
    random.shuffle(c)

    images, masks = zip(*c)

    images = list(images)
    masks = list(masks)

    # Preview training images and their corresponding annotations
    if CHECK_IMAGES:
        preview_images(images, masks)

    # Start training process
    train(images, masks)

    
    