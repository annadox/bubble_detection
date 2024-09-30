from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

def estimate_noise(I):

    print(I.shape)

    H, W = I.shape

    M = [[1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]]

    sigma = np.sum(np.sum(np.absolute(convolve2d(I, M))))
    sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W-2) * (H-2))

    return sigma

def fft_image(gray_image):
    dft = cv2.dft(np.float32(gray_image), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    return magnitude_spectrum

# Create a Tkinter root window (it won't be shown)
root = Tk()
root.withdraw()  # Hide the Tkinter root window

# Open file dialog to select an image file
file_path1 = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

file_path2 = askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")])

if file_path1 and file_path2:
    # Read the image using OpenCV
    image1 = cv2.imread(file_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(file_path2, cv2.IMREAD_GRAYSCALE)

    s1 = estimate_noise(image1)
    s2 = estimate_noise(image2)

    print(s1, s2)

    spectrum = [fft_image(image1), fft_image(image2)]

    fig = plt.figure(figsize=(15, 5))

    # Add subplots to the figure
    for i in range(len(spectrum)):
        ax = fig.add_subplot(1, len(spectrum), i+1)
        ax.imshow(spectrum[i], cmap='gray')
        ax.set_title(f"Spectrum of image{i + 1}")
        ax.axis('off')

    cv2.imshow("image1", image1)
    cv2.imshow("image2", image2)        

    cv2.waitKey() # Press key to display the fft plots
    plt.show()

        


        