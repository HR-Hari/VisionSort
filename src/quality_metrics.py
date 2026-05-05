import cv2
import numpy as np

def compute_sharpness (image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray,cv2.CV_64F)
    return laplacian.var()

def compute_brightness(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return np.mean (gray)

def compute_contrast(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return np.std (gray)

