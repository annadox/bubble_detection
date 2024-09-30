import cv2
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

# This script is used to create images that only display part of the bubbles in the image (to even out the representation of different types of bubbles in the datataset).
# It creates horizontal lines at a height "i" and j then replaces the part of the image outside of them with its background
# Move the lines up or down with 'W' and 'S', switch between top and bottom line with 'M' end the process and save with 'Q'
# Minimum band height (position difference between two lines is currently set to 50px)


IMAGE_PATH = None
BACKGROUND_PATH = None

SAVE_DIR = "representation_images"

MIN_BAND_HEIGHT = 50 # Change to desired value

if __name__ == "__main__":
    if IMAGE_PATH == None:
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    else:
        image_path = IMAGE_PATH

    if BACKGROUND_PATH == None:
        root = tk.Tk()
        root.withdraw()
        background_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    else:
        background_path = BACKGROUND_PATH
    
    image = cv2.imread(image_path)
    background = cv2.imread(background_path)

    end = False
    changing = "top line"
    i = 0
    j = 800
    while not end:
        image_cpy = np.copy(image)
        image[0:i] = background[0:i]
        image[j:image.shape[1]] = background[j:image.shape[1]]

        print(f"top line:{i}px, bottom line:{j}px,", f"changing: {changing}")
        cv2.imshow("image", image)

        key = cv2.waitKey(0)
        if key == ord('q'):
            end = True
            break

        image[0:i] = image_cpy[0:i]
        image[j:image.shape[1]] = image_cpy[j:image.shape[1]]
        if changing == "top line":
            if i < 799 and (j - i > MIN_BAND_HEIGHT) and key == ord("s"):
                i += 1
            elif i > 0 and key == ord("w"):
                i -= 1

        elif changing == "bottom line":
            if j < 800 and key == ord("s"):
                j += 1
            elif j > 0 and (j - i > MIN_BAND_HEIGHT) and key == ord("w"):
                j -= 1

        if key == ord('m'):
            if changing == "top line":
                changing = "bottom line"
            else:
                changing = "top line"

    cv2.imwrite(os.path.join(SAVE_DIR, f"{os.path.splitext(os.path.basename(image_path))[0]}_merged.png"), image)