import numpy as np
import cv2
import pandas as pd

# Here we define some of assisting functions that are used throught the project.

# This class is used to read the .raww images.
class RawwReader:

    def __init__(self, img_size=(800, 1280), rotation=None): # img_size is in format (y, x)
        self.img_size = img_size
        self.rotation = rotation

    def read_raw_image(self, file_path):
        # Read image from file
        raw_img = np.fromfile(file_path, dtype='int16', sep="")
        
        # Reshape it to size
        raw_img = np.reshape(raw_img, self.img_size) 
        
        # Rotate if needed
        if not self.rotation is None:
            raw_img = cv2.rotate(raw_img, self.rotation)

        # Normalize the image
        raw_img = cv2.normalize(raw_img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return raw_img

# Draws bubbles on an image from its corresponding labels
def draw_bubbles(img, bubbles):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for b in bubbles:
        img = cv2.ellipse(img, (int(b[1]), int(b[0])), (int(b[4]), int(b[3])), int(np.degrees(b[2])), startAngle = 0, endAngle = 360, color=(0,255,0), thickness=1)

    return img

# Draws bubbles from datafram with the possiblity of selecting a color of the annotation
def draw_bubbles_color_df(img, bubbles, clr):
    for index, row in bubbles.iterrows():
        img = cv2.ellipse(img, (int(row["centerX"]), int(row["centerY"])), (int(row["minor_axis_length"]), int(row["major_axis_length"])), int(np.degrees(row["orientation"])), startAngle=0, endAngle=360, color=clr, thickness=1) 

    return img

# Draws bubbles from dataframe
def draw_bubbles_dataframe(img, bubbles):
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for index, row in bubbles.iterrows():
        img = cv2.ellipse(img, (int(row["centerX"]), int(row["centerY"])), (int(row["major_axis_length"]), int(row["minor_axis_length"])), int(np.degrees(row["orientation"])), startAngle=0, endAngle=360, color=(0,255,0), thickness=1) 

    return img

if __name__ == "__main__":
    IMAGE = "training_dataset/images/10.png"
    LABEL = "training_dataset/csv_data/10.csv"
    img = cv2.imread(IMAGE)
    df = pd.read_csv(LABEL)

    img = draw_bubbles_dataframe(img, df)
    
    cv2.imshow("1", img)
    cv2.waitKey()
