# pylint:disable=no-member

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#print(tess.get_languages(config=''))

txt = 'Hello path effects world!This is the normal '
img = cv.imread(r'resources\rg_83-1_v_hor.jpg')
conv_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#dst = cv.fastNlMeansDenoisingColored(img,None,10,10,7,21)
dst = cv.fastNlMeansDenoising(conv_img,None,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)

plt.figtext(0.5, 0.1, txt, wrap=True, horizontalalignment='center', fontsize=12)
plt.show()