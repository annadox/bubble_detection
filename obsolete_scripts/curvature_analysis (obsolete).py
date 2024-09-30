from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def high_distance_points(p1, p2, t):
    manhattan = 0
    for i, j in zip(p1, p2):
        manhattan += abs(i - j)

    if manhattan > t:
        return True
    
    return False

def fit_parabola(height):
        sigma = 2.0  # Standard deviation for Gaussian kernel
        smooth_col = gaussian_filter1d(height, sigma)
        x_val = np.asarray(range(len(height)))

        mean = np.mean(smooth_col)
        std = np.std(smooth_col)

        # Define the range to filter out extreme values (within two standard deviations from the mean)
        lower_bound = mean - 2 * std
        upper_bound = mean + 2 * std

        mask = (smooth_col >= lower_bound) & (smooth_col <= upper_bound)
        x_filtered = x_val[mask]
        smooth_col_filtered = smooth_col[mask]

        coefficients = np.polyfit(x_filtered, smooth_col_filtered, 2)
        polynomial = np.poly1d(coefficients)
        y_values = polynomial(x_val)

        return y_values, coefficients

# Create a Tkinter root window (it won't be shown)
root = Tk()
root.withdraw()  # Hide the Tkinter root window

# Open file dialog to select an image file
file_path = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

if file_path:
    # Read the image using OpenCV
    image1 = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    ret, thresh1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    
    blurred = cv2.GaussianBlur(image1, (3, 3), sigmaX=0, sigmaY=0)
    can = cv2.Canny(blurred, threshold1=40, threshold2=80)

    cv2.imshow("canny", can)      
    cv2.imshow("i", image1) 
    #cv2.waitKey()

    #lines = cv2.HoughLinesP(can, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)

    contours, _ = cv2.findContours(can, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_lengths = [(cv2.arcLength(contour, False), contour) for contour in contours]
    sc = sorted(contour_lengths, key=lambda x: x[0], reverse=True)

    # Draw the two longest contours on the original image
    """ if len(sorted_contours) > 0:
        for i in range(len(sorted_contours)):
            if (cv2.arcLength(sorted_contours[i][1], False) > 20):
                cv2.drawContours(image1, [sorted_contours[i][1]], -1, (0, 255, 0), 2) """

    empty = np.zeros_like(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)

    sorted_contours = []
    for i in range(len(sc)):
        current_contour = sc[i][1]
        if cv2.contourArea(current_contour) *3 < cv2.arcLength(current_contour, True):
            sorted_contours.append(sc[i])
            cv2.drawContours(empty, [current_contour], -1, (255, 255, 255), 1)
    
    cv2.imshow("ety", empty)       
    cv2.waitKey()

    end_points = []
    for i in range(len(sorted_contours)):
        current_contour = sorted_contours[i][1]
        eps = [(np.inf,np.inf), (0,0)]
        for point in current_contour:
            if point[0][0] < eps[0][0]:
                eps[0] = list(point[0])

            if point[0][0] > eps[1][0]:
                eps[1] = list(point[0])

        end_points.append(eps)

    print(list(zip(*end_points))[0])
    print(list(zip(*end_points))[1])
    for i, p1 in enumerate(list(zip(*end_points))[0]):
        print(f"\r", i, len(end_points), end="")
        min_distance = float('inf')
        closest_point = None
        cpt_index = 0
        for j, p2 in enumerate(list(zip(*end_points))[1]):
            distance = np.linalg.norm(np.asarray(p1) - np.asarray(p2))
            if distance < min_distance:
                min_distance = distance
                closest_point = p2
                cpt_index = j

        cpy = np.copy(image1)
        if (min_distance < 20):   
            cv2.drawContours(cpy, [sorted_contours[i][1], sorted_contours[cpt_index][1]], -1, (0, 255, 0), 2)
            cv2.circle(cpy, closest_point, 3, (0,0,255), -1)
            cv2.circle(cpy, p1, 3, (255,0,0), -1)
            cv2.line(empty, closest_point, p1, (255,255,255), 1)

        """ cv2.imshow("image", cpy)
        cv2.waitKey() """

    cv2.imshow("canny", empty)       
    cv2.waitKey()
    contours, _ = cv2.findContours(empty, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_lengths = [(cv2.arcLength(contour, False), contour) for contour in contours]
    sorted_contours = sorted(contour_lengths, key=lambda x: x[0], reverse=True)

    process_image = np.zeros_like(empty)

    if len(sorted_contours) > 0:
        for i in range(min(2, len(sorted_contours))):
            cv2.drawContours(process_image, [sorted_contours[i][1]], -1, (255, 255, 255), 1)

    cv2.imshow("process", process_image)

    top_height = []
    bottom_height = []

    for col in range(process_image.shape[1]):
        top_c = 0
        bottom_c = 0
        for row in range(process_image.shape[0]):
            if process_image[row, col] == 0:
                top_c += 1
            elif process_image[row, col] == 255:
                break
        for row in reversed(range(process_image.shape[0])):
            if process_image[row, col] == 0:
                bottom_c += 1
            elif process_image[row, col] == 255:
                break
        
        cc = process_image.shape[0] - top_c - bottom_c
        top_height.append(top_c)
        bottom_height.append(process_image.shape[0] - bottom_c)

    y_top, coef_top = fit_parabola(top_height)
    y_bottom, coef_bottom = fit_parabola(bottom_height)

    plt.gca().invert_yaxis()
    plt.plot(top_height, color="blue")
    plt.plot(y_top, color="red")
    plt.plot(bottom_height, color="blue")
    plt.plot(y_bottom, color="red")
    plt.draw() 

    for i in range(len(y_top)):
        image1[int(y_top[i]), i] = (0, 255, 0)
        image1[int(y_bottom[i]), i] = (0, 255, 0)

    cv2.imshow('parabolas', image1)
  
    cv2.waitKey()
    plt.show()