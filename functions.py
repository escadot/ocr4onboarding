# pylint:disable=no-member

try:
    from PIL import Image
except ImportError:
    import Image
import cv2
import numpy as np
import pytesseract as tess
from pytesseract import Output
from matplotlib import pyplot as plt

tess.pytesseract.tesseract_cmd=r'%LocalAppData%\Programs\Tesseract-OCR\tesseract'

def set_input():
    from tkinter import Tk as tk
    from tkinter.filedialog import askopenfilename

    tk().withdraw()
    inputImage = askopenfilename() # show an "Open" dialog box and return the path to the selected file
    return cv2.imread(inputImage), cv2.imread('resources/template.jpg')


# Função não utilizada
# def GetFurtherSidesValues(_mask):
#     i = 0
#     j = 0
#     uppervalue = 0
#     lowervalue = 0
#     leftvalue = 0
#     rightvalue = 0
#     for linha in _mask:
#         i += 1
#         j = 0
#         for coluna in linha:
#             if coluna:
#                 uppervalue = i
#                 break
#             j += 1
#         if uppervalue != 0:
#             break
#     i = 0
#     j = 0
#     for linha in reversed(_mask):
#         i += 1
#         j = 0
#         for coluna in reversed(linha):
#             if coluna:
#                 lowervalue = len(_mask[0]) - i
#                 break
#             j += 1
#         if lowervalue != 0:
#             break
#     for coluna in range(len(_mask[0])):
#         linha = 0
#         for linha in range(len(_mask)):
#             if _mask[linha, coluna]:
#                 leftvalue = coluna
#                 break
#         if leftvalue != 0:
#             break
#     for coluna in range(len(_mask[0])):
#         novaColuna = len(_mask[0]) - coluna - 1
#         for linha in range(len(_mask)):
#             novaLinha = len(_mask) - linha - 1
#             if _mask[novaLinha, novaColuna]:
#                 rightvalue = novaColuna
#                 break
#         if rightvalue != 0:
#             break
#     return (uppervalue, lowervalue, leftvalue, rightvalue, )

def noise_removal(image):
    return cv2.fastNlMeansDenoisingColored(image,None,6,5,9)

def resize_image(image):
    return cv2.resize(image, (720, 476),
                             interpolation=cv2.INTER_AREA)

# thresholding
def thresholding_otsu(image):
    return cv2.threshold(image, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

def thresholding_gaussian(image):
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def clean_image(image):
    return cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
    

def deskew(image):
    # preprocessing
    gray = get_grayscale(image)
    gray = cv2.bitwise_not(gray)

    thresh = thresholding_otsu(gray)

    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle
    
    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated

def show_boxes(image, confidence):
    from copy import deepcopy

    img = deepcopy(image)
    d = tess.image_to_data(img, lang='por', output_type=Output.DICT)

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if float(d['conf'][i]) > confidence:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img

def select_roi(image):
    # seleciona o ROI
    myroi = cv2.selectROI("Selecione a ROI", image, False, False)
    # corta a imagem
    image = image[int(myroi[1]):int(myroi[1]+myroi[3]),
                int(myroi[0]):int(myroi[0]+myroi[2])]

    # mostra a imagem cortada
    cv2.imshow("ROI", image)
    cv2.waitKey(0)
    # cv2.imwrite("imgNome.jpg", output)

    return image

def show_img(image):
    cv2.imshow("Imagem", image)
    cv2.waitKey(0)


def img2text(image):
    return tess.image_to_string(image, lang='por')

def plot_result(inputImage, txt_box, text):
    plt.subplot(121),plt.imshow(inputImage)    
    plt.title('Original'),plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(txt_box, cmap='gray', vmin=0, vmax=255)
    plt.title('Text boxes'),plt.xticks([]), plt.yticks([])

    plt.figure()
    plt.text(0, 0, text, ha='left')
    plt.axis('off'), plt.xticks([]), plt.yticks([])

    plt.show()
