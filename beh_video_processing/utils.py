import os
import glob
import numpy as np
import cv2
from skimage import draw
import matplotlib.pyplot as plt


def check_template_matching(img, coordinates, show=False):
    """
    check_template_matching draws the rectangles found with
    your rectangle annotation method in order to verify
    if they were found correctly.

    Args:
        img (np.array()): image of the experiment. Typically
                          the mean well image but it can be
                          any frame of your video.
        coordinates (np.array() of shape (:,4)): array with n rows
                        as the number of rectangles and 4 columns
                        givin x (dim[0]), y (dim[1]), width (dim[2])
                        and height (dim[3]) of the rectangle,
                        in this order.

    Returns:
        img: the image with the rectangles drawn in white.
    """
    for rectangle in coordinates:
        rect = draw.rectangle_perimeter(start=(rectangle[1], rectangle[0]),
                                        extent=(rectangle[3], rectangle[2]))
        img[rect[1], rect[0]] = 255

    if show:
        plt.imshow(img)
        plt.show()

    return img


def get_mean_well_img(path):
    """Compute the mean image from all png images in a folder.
        The mean image is saved in the same folder.

    Args:
        path (string): folder path to directory where well images are located
    """

    # Once we have the well images cropped from the background images
    # we can create an average well image template
    images = glob.glob(path + r'\*png')

    count = 0
    for well in images:
        single_well = cv2.imread(well)
        if count == 0:
            w, h, _ = single_well.shape
            all_wells = np.empty([len(images), w, h])
        single_well = cv2.cvtColor(single_well, cv2.COLOR_BGR2GRAY)
        all_wells[count, :, :] = single_well
        count += 1

    # compute mean well image and store it as np.uint8
    mean_well = np.mean(all_wells, axis=0)
    mean_well = mean_well.astype(np.uint8)

    # save mean well image
    cv2.imwrite(os.path.join(path, r'mean_well.png'), mean_well)
