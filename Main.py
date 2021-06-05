# pylint:disable=no-member

import functions

def main():
    inputImage, template = functions.set_input()

    functions.show_img(inputImage)

    output = functions.noise_removal(inputImage)

    # ajusta o input para um valor mais visível
    output = functions.resize_image(output)

    output = functions.deskew(output)

    functions.show_img(output)

    output = functions.clean_image(output)

    output = functions.get_grayscale(output)
    # output = functions.thresholding_gaussian(output)
    # output = functions.thresholding_otsu(output)

    functions.show_img(output)

 

    # txt_boxes = functions.show_boxes(output, 80)

    # output = functions.select_roi(output)

    text = functions.img2text(output)

    functions.plot_result(inputImage, output, text)
    # print(text)

if __name__ == "__main__":
    main()