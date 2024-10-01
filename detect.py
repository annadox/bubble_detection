from csbdeep.utils import normalize
from stardist.models import StarDist2D
import numpy as np
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.rcParams["image.interpolation"] = None
import matplotlib.pyplot as plt
from skimage import measure
import pandas as pd
from stardist import random_label_cmap
import os
import cv2
from utils import RawwReader
from utils import draw_bubbles
import tifffile

# This script detects the bubbles on a selected directory of measurements.

# Change the constants in this script to your needs before running it.
# The directory at DATA_DIR should be structured as follows:
# each set of measurements should have their own directory (MERITVE) in the data directory, 
# then each measurement in should have its own sub-directory where the images of the measurement are located.

ROOT_DIR = os.getcwd() # Root directory of the script file
MERITVE = "Example measurements Vertical" # Sub-directory of the set of mesurements you want to detect (needs to contain other sub-directories for each set of images)
DATA_DIR = "./process_data"
MEASUREMENTS_DIR = os.path.join(DATA_DIR, MERITVE) # Sets the sub-directory

OUTPUT_DIR = os.path.join(ROOT_DIR, "output", MERITVE) # Output dir for the predictions

# Create output dir if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

# Since we are predicting from the .raww images, where the rod is vertical and the flow of the coolant is upwards (height=1280, width=800),
# we need to rotate them counterclockwise to make the rod vertical and the coolant flow to the right (height=800, width=1280).
# Make sure that all images in the MEASUREMENTS_DIR are the same dimensions

IMG_SIZE = (1280, 800) # Set this to the size of the RAWW input image (y, x).
IMAGE_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE # Set depending on how you want the input image to be rotated - set to None for no rotation.

DISPLAY = False # Displays the predictions on the image if set to True. (Unpractical for large measurement datasets, use just for testing)

PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "csv_data") # Path to the predictions csv_data directory in the output directory.
PREDICTIONS_IMG_AMT = -1 # Set to how many images from each measurement you want predicted - set to -1 if you want all images.

MASKS_DIR = os.path.join(OUTPUT_DIR, "masks") # Path to the predictions masks (.tif) directory in the output directory.
SAVE_MASKS = True # Set to True if you want masks saved to the masks directory.

IMAGES_DIR = os.path.join(OUTPUT_DIR, "images") # Path to the .png format images of the predicted images.
SAVE_IMGS = True # Set True if you want images saved to a directory

EXAMPLE_IMG_DIR = os.path.join(OUTPUT_DIR, "example_images") # Path to the directory with images, which have predictions annotated on them.
EXAMPLE_IMG_AMT = -1 # Set to how many example images you want saved.

MODEL_BASE_DIR = "models/full_model_best"

test_section_bounds = [0, 800] # This variable stores the pixel height interval of the test section, anyt detection outside of this will be discarded.

# Reads test_section_bounds from 
section_path = os.path.join(MEASUREMENTS_DIR, "test_section_bounds.txt")

if os.path.exists(section_path):
    with open(section_path, 'r') as file:
        for line in file:
            line = line.strip()
            numbers = line.split(',')
            test_section_bounds = [int(num) for num in numbers]

# Predict single images
def predict_single_img(model, image, name):
    current_img_bubbles = []

    # Predict image with model - returns a mask with 0 where there is no bubble and the number of the bubble where there is one.
    pred,_ = model.predict_instances(image)

    # Remove all predictions out of the test section height interval
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if i < test_section_bounds[0] or i > test_section_bounds[1]:
                pred[i,j] = 0

    # Create elipses from each detection (for each blob of the same number on the mask).
    props = measure.regionprops_table(
        pred, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length')
    )
    for cy, cx, ori, major, minor in zip(props.get('centroid-0'), props.get('centroid-1'), props.get('orientation'), props.get('major_axis_length'), props.get('minor_axis_length')):
        current_img_bubbles.append([cy, cx, ori, major/2, minor/2])

    # Display predicted images if option is true
    if(DISPLAY):
        display_single(image, name, current_img_bubbles)

    return pred, current_img_bubbles

# Predict from single measurement directroy
def predict_from_dir(model, dir_path):
    # Returns an dictionary with keys "masks", "names", "images", "bubbles".
    predictions = {
            "masks": [], # All predicted mask images in the current dir
            "names": [], # Image file names without the file extension
            "images": [], # Input images from the current dir
            "bubbles": [] # Labels of the predicted bubbles
        }

    dir_list = os.listdir(dir_path)

    if len(dir_list) < PREDICTIONS_IMG_AMT:
        raise ValueError("PREDICTIONS_IMG_AMT is larger than the amount of images in a directory")
    elif PREDICTIONS_IMG_AMT == 0 or PREDICTIONS_IMG_AMT == -1:
        step_preds = 1
    else:
        step_preds = len(dir_list)//PREDICTIONS_IMG_AMT

    # Initialize rawwreader
    rr = RawwReader(IMG_SIZE, IMAGE_ROTATION)

    # Read each image and append its predictions to the dictionary.
    for img in tqdm(dir_list[::step_preds]):
        raw_img = rr.read_raw_image(os.path.join(dir_path, img))
        img_norm = normalize(raw_img, 1, 99.8, axis=(0,1))

        pred, current_img_bubbles = predict_single_img(model, img_norm, img.rsplit(".")[0])

        predictions["bubbles"].append(current_img_bubbles)
        predictions["names"].append(img.rsplit(".")[0])
        predictions["masks"].append(pred)
        predictions["images"].append(raw_img)

    return predictions

# Runs prediction for every sub-directory (measurement) in data directory
def run_dirs(model, dataset_dir):
    for file in os.listdir(dataset_dir)[0:]:
        print("\nProcessing: " + file)
        if os.path.isdir(os.path.join(dataset_dir,file)):
            current_dir = os.path.join(dataset_dir, file)
            predictions = predict_from_dir(model, current_dir)

            # Create directories for the set of measurements
            curr_exp_dir = os.path.join(EXAMPLE_IMG_DIR, file)
            if not os.path.exists(curr_exp_dir):
                os.mkdir(curr_exp_dir)

            curr_pred_dir = os.path.join(PREDICTIONS_DIR, file)
            if not os.path.exists(curr_pred_dir):
                os.mkdir(curr_pred_dir)

            curr_mask_dir = os.path.join(MASKS_DIR, file)
            if not os.path.exists(curr_mask_dir):
                os.mkdir(curr_mask_dir)

            curr_image_dir = os.path.join(IMAGES_DIR, file)
            if not os.path.exists(curr_image_dir):
                os.mkdir(curr_image_dir)
            
            # Write .csv files of predictions and save example images from the current measurement
            write_predictions_csv(predictions, curr_pred_dir)
            write_example_images(predictions, curr_exp_dir)

            # Save masks and images from the current measurement
            if SAVE_MASKS:
                write_masks(predictions, curr_mask_dir)
            if SAVE_IMGS:
                write_imgs(predictions, curr_image_dir)

# Write predictions to csv_data  directory (.csv)
def write_predictions_csv(predictions, dir):
    bubbles = predictions["bubbles"]
    names = predictions["names"]

    for i in range(len(bubbles)):
        df = pd.DataFrame(bubbles[i] ,columns=['centerY', 'centerX', 'orientation', 'major_axis_length', "minor_axis_length"])
        df = df.round(3)

        img_name = names[i]

        df.to_csv(dir + "/" + img_name + ".csv", index=False)

    print("Predictions written to: " + dir)

# Write masks to masks directory (.tif)
def write_masks(predictions, dir):
    masks = predictions["masks"]
    names = predictions["names"]

    for mask, name in zip(masks,names):
        mask = mask.astype(np.uint16)
        tifffile.imwrite(os.path.join(dir, name + ".tif"), mask)

    print("Masks written to: " + dir)

# Write images to images folder (.png)
def write_imgs(predictions, dir):
    imgs = predictions["images"]
    names = predictions["names"]

    for img, name in zip(imgs, names):
        cv2.imwrite(os.path.join(dir, name + ".png"), img)

    print("Images written to: " + dir)

# Write example images to example_images folder (.png)
def write_example_images(predictions, dir):
    imgs = predictions["images"]
    bubbles = predictions["bubbles"]
    names = predictions["names"]

    if len(imgs) < EXAMPLE_IMG_AMT:
        raise ValueError("EXAMPLE_IMG_AMT is larger than the amount of images in a directory")
    elif EXAMPLE_IMG_AMT == 0:
        return
    elif EXAMPLE_IMG_AMT == -1:
        step_examples = 1
    else:
        step_examples = len(imgs)//EXAMPLE_IMG_AMT + 1

    for img, current_img_bubbles, name in list(zip(imgs, bubbles, names))[::step_examples]:
        img = draw_bubbles(img, current_img_bubbles)

        cv2.imwrite(os.path.join(dir, name + ".jpg"), img)

    print("Example images written to: " + dir)

# Display single image with annotated bubble labels
def display_single(img, name, bubbles):
    img = draw_bubbles(img, bubbles)    
    
    cv2.imshow(name + ".jpg", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":

    # Create dirs in the OUTPUT_DIR for every different output
    if not os.path.exists(PREDICTIONS_DIR):
        os.mkdir(PREDICTIONS_DIR)
    
    if not os.path.exists(EXAMPLE_IMG_DIR):
        os.mkdir(EXAMPLE_IMG_DIR)

    if not os.path.exists(MASKS_DIR):
        os.mkdir(MASKS_DIR)
    
    if not os.path.exists(IMAGES_DIR):
        os.mkdir(IMAGES_DIR)

    # Create StarDist2D model
    strdist = StarDist2D(None, name='stardist', basedir=MODEL_BASE_DIR)

    run_dirs(strdist, MEASUREMENTS_DIR)
    

    