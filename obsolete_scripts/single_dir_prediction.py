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


""" The code will automatically create three subfolders in the OUTPUT_DIR:
    - 'csv_data': parameters of ellipses that describe the bubbles
    - 'masks': colored masks of the predictions
    - 'example_images': images with the predictions drawn over the original image """

ROOT_DIR = os.getcwd()
MERITEV = "Boiling meritve Vertical"
FOLDER = "Recording_Date=240320_Time=134257_40C_300kgm2s_p1"

OUTPUT_DIR = os.path.join(ROOT_DIR, "single_output", MERITEV + "+" + FOLDER)

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

""" Create subfolders in the DATA_DIR
    for each set of measurements. (even if you only have one measurement set)
    """

DATA_DIR = os.path.join("../data/", MERITEV, FOLDER) #"../data/Boiling meritve Vertical"
IMG_SIZE = (1280, 800) # Set this to the size of the RAWW input image (y, x)
IMAGE_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE # Set depending on how you want the input image to be rotated - set to None for no rotation

DISPLAY = False # Displays the predictions on the image if set to True

PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "csv_data")
PREDICTIONS_IMG_AMT = 50 # Set to how many images from each measurement you want predicted - set to 0 if you want all images

MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
SAVE_MASKS = True # Set to True if you want masks saved to the masks directory

IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
SAVE_IMGS = True
FIRST = 0
LAST = 400

EXAMPLE_IMG_DIR = os.path.join(OUTPUT_DIR, "example_images")

def write_predictions_csv(predictions, dir):
    bubbles = predictions["bubbles"]
    names = predictions["names"]

    for i in range(len(bubbles)):
        df = pd.DataFrame(bubbles[i] ,columns=['centerY', 'centerX', 'orientation', 'major_axis_length', "minor_axis_length"])
        df = df.round(3)

        img_name = names[i]

        df.to_csv(dir + "/" + img_name + ".csv", index=False)

    print("Predictions written to: " + dir)

def write_masks(predictions, dir):
    masks = predictions["masks"]
    names = predictions["names"]

    for mask, name in zip(masks,names):
        mask = mask.astype(np.uint16)
        tifffile.imwrite(os.path.join(dir, name + ".tif"), mask)

    print("Masks written to: " + dir)

def write_imgs(predictions, dir):
    imgs = predictions["images"]
    names = predictions["names"]

    for img, name in zip(imgs, names):
        cv2.imwrite(os.path.join(dir, name + ".png"), img)

    print("Images written to: " + dir)

def write_example_images(predictions, dir):
    imgs = predictions["images"]
    bubbles = predictions["bubbles"]
    names = predictions["names"]

    for img, current_img_bubbles, name in list(zip(imgs, bubbles, names)):
        img = draw_bubbles(img, current_img_bubbles)

        cv2.imwrite(os.path.join(dir, name + ".jpg"), img)

    print("Example images written to: " + dir)

def predict_single_img(model, image, name):
    current_img_bubbles = []
    num_bubbles = 1
    preds = []

    pred,_ = model.predict_instances(image)

    props = measure.regionprops_table(
        pred, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length')
    )
    for cy, cx, ori, major, minor in zip(props.get('centroid-0'), props.get('centroid-1'), props.get('orientation'), props.get('major_axis_length'), props.get('minor_axis_length')):
        current_img_bubbles.append([cy, cx, ori, major/2, minor/2])

    return pred, current_img_bubbles

def predict_from_dir(model, dir_path):
    predictions = {
        "masks": [],
        "names": [],
        "images": [],
        "bubbles": []
    }

    dir_list = os.listdir(dir_path)

    if len(dir_list) < PREDICTIONS_IMG_AMT:
        raise ValueError("PREDICTIONS_IMG_AMT is larger than the amount of images in a directory")
    elif PREDICTIONS_IMG_AMT == 0:
        step_preds = 1
    else:
        step_preds = len(dir_list)//PREDICTIONS_IMG_AMT

    rr = RawwReader(IMG_SIZE, IMAGE_ROTATION)

    for img in tqdm(dir_list[0:400:1]):
        raw_img = rr.read_raw_image(os.path.join(dir_path, img))
        img_norm = normalize(raw_img, 1, 99.8, axis=(0,1))

        pred, current_img_bubbles = predict_single_img(model, img_norm, img.rsplit(".")[0])

        predictions["bubbles"].append(current_img_bubbles)
        predictions["names"].append(img.rsplit(".")[0])
        predictions["masks"].append(pred)
        predictions["images"].append(raw_img)

    return predictions
    
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
    strdist = StarDist2D(None, name='stardist', basedir='models/full_model_best')

    predictions = predict_from_dir(strdist, DATA_DIR)
    write_predictions_csv(predictions, PREDICTIONS_DIR)
    write_example_images(predictions, EXAMPLE_IMG_DIR)

    if SAVE_MASKS:
        write_masks(predictions, MASKS_DIR)
    if SAVE_IMGS:
        write_imgs(predictions, IMAGES_DIR)

    