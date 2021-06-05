# pylint:disable=no-member

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#print(tess.get_languages(config=''))

# get grayscale image
def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv.medianBlur(image, 5)

# thresholding
def thresholding(image, type):
    return cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

img = cv.imread('resources/img_thresh.jpg')

blur = cv.GaussianBlur(img,(3,3), cv.BORDER_DEFAULT)
denoise = cv.fastNlMeansDenoising(blur,None,10,7,21)

plt.subplot(131),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(blur),plt.title('Gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(denoise),plt.title('Denoise')
plt.xticks([]), plt.yticks([])

plt.show()