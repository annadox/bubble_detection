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

""" The code will automatically create three subfolders in the OUTPUT_DIR:
    - 'csv_data': parameters of ellipses that describe the bubbles
    - 'masks': colored masks of the predictions
    - 'example_images': images with the predictions drawn over the original image """

ROOT_DIR = os.getcwd()
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

""" Create subfolders in the DATA_DIR
    for each set of measurements. (even if you only have one measurement set)
    """

DATA_DIR = "../data/Boiling meritve Vertical"
IMG_SIZE = (1280, 800) # Set this to the size of the RAWW input image (y, x)
IMAGE_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE # Set depending on how you want the input image to be rotated - set to None for no rotation

DISPLAY = False # Displays the predictions on the image if set to True

PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "csv_data")
PREDICTIONS_IMG_AMT = 50 # Set to how many images from each measurement you want predicted - set to 0 if you want all images

MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")
SAVE_MASKS = False # Set to True if you want masks saved to the masks directory (not recomended for multiple images as it takes a REALLY long time)

IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
SAVE_IMGS = True

EXAMPLE_IMG_DIR = os.path.join(ROOT_DIR, "output/example_images")
EXAMPLE_IMG_AMT = 4 # Set to how many example images you want saved

PIPELINE_OVERLAP_THRESHOLD = 0.25 # This decides how much a detection can overlap with detections, from previous models in the pipeline, before its discarded


def run_dirs(model_pipeline, dataset_dir):
    for file in os.listdir(dataset_dir)[0:]:
        print("\nProcessing: " + file)
        if os.path.isdir(os.path.join(dataset_dir,file)):
            current_dir = os.path.join(dataset_dir, file)
            predictions = predict_from_dir(model_pipeline, current_dir)

            # Create directories for the set of measurements
            curr_exp_dir = os.path.join(EXAMPLE_IMG_DIR, file)
            if not os.path.exists(curr_exp_dir):
                os.mkdir(curr_exp_dir)

            curr_pred_dir = os.path.join(PREDICTIONS_DIR, file)
            if not os.path.exists(curr_pred_dir):
                os.mkdir(curr_pred_dir)

            if SAVE_MASKS:
                curr_mask_dir = os.path.join(MASKS_DIR, file)
                if not os.path.exists(curr_mask_dir):
                    os.mkdir(curr_mask_dir)

            if SAVE_IMGS:
                curr_image_dir = os.path.join(IMAGES_DIR, file)
                if not os.path.exists(curr_image_dir):
                    os.mkdir(curr_image_dir)
            
            write_predictions_csv(predictions, curr_pred_dir)
            write_example_images(predictions, curr_exp_dir)

            if SAVE_MASKS:
                write_masks(predictions, curr_mask_dir)
            if SAVE_IMGS:
                write_imgs(predictions, curr_image_dir)

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
        lbl_cmap = random_label_cmap()
        plt.imsave(os.path.join(dir, name + ".png"), mask, cmap=lbl_cmap)

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

    if len(imgs) < EXAMPLE_IMG_AMT:
        raise ValueError("EXAMPLE_IMG_AMT is larger than the amount of images in a directory")
    elif EXAMPLE_IMG_AMT == 0:
        return
    else:
        step_examples = len(imgs)//EXAMPLE_IMG_AMT + 1

    for img, current_img_bubbles, name in list(zip(imgs, bubbles, names))[::step_examples]:
        img = draw_bubbles(img, current_img_bubbles)

        cv2.imwrite(os.path.join(dir, name + ".jpg"), img)

    print("Example images written to: " + dir)

def predict_single_img(model_pipeline, image, name):
    image_copy = image.copy()
    current_img_bubbles = []
    num_bubbles = 0
    preds = np.zeros_like(image)

    # Predict with all models in the pipeline
    for model in model_pipeline:
        pred,_ = model.predict_instances(image_copy)

        uniq_bub = np.unique(pred.ravel())

        # If the model finds bubbles already found by the previous models this code checks for overlap and removes it if it overlaps more than a certain threshold
        for num in uniq_bub[1:]:
            overlap = 0
            inds = list(zip(*np.where(pred == num)))
            for (i, j) in inds:
                if preds[i,j] != 0:
                    overlap += 1

            if overlap/len(inds) >= PIPELINE_OVERLAP_THRESHOLD:
                pred[pred == num] = 0
        
        # Adds new predictions to the predictions of the previous models in the pipeline
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if(pred[i,j] != 0):
                    pred[i,j] += num_bubbles

                if(preds[i,j] == 0):
                    preds[i,j] = pred[i,j]

        # Gets number of bubbles for labels in next model prediction
        num_bubbles += max(pred.ravel())

        # Blanks out all the found bubbles        
        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                if pred[i,j] != 0 :
                    image_copy[i, j] = 1

    preds = preds.astype(int)
    
    # Extract region properties from prediction masks
    props = measure.regionprops_table(
        preds, properties=('centroid', 'orientation', 'major_axis_length', 'minor_axis_length')
    )
    for cy, cx, ori, major, minor in zip(props.get('centroid-0'), props.get('centroid-1'), props.get('orientation'), props.get('major_axis_length'), props.get('minor_axis_length')):
        current_img_bubbles.append([cy, cx, ori, major/2, minor/2])

    # Display the image if the option is marked True
    if(DISPLAY):
        display_single(image, name, current_img_bubbles)

    return pred, current_img_bubbles

def predict_from_dir(model_pipeline, dir_path):
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

    for img in tqdm(dir_list[::step_preds]):
        raw_img = rr.read_raw_image(os.path.join(dir_path, img))
        img_norm = normalize(raw_img, 1, 99.8,axis=(0,1))

        pred, current_img_bubbles = predict_single_img(model_pipeline, img_norm, img.rsplit(".")[0])

        predictions["bubbles"].append(current_img_bubbles)
        predictions["names"].append(img.rsplit(".")[0])
        predictions["masks"].append(pred)
        predictions["images"].append(raw_img)

    return predictions

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
    small_model = StarDist2D(None, name='stardist', basedir='models/small_bubbles')
    medium_model = StarDist2D(None, name='stardist', basedir='models/full_model_best')

    model_pipeline = [medium_model, small_model]

    run_dirs(model_pipeline, DATA_DIR)
    

    