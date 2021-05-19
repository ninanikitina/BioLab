import os
import glob
import csv
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def get_3d_image(input_folder, type):
    object_layers = []

    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + type + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])

        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layers.append([object_img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    nucleus_3d_image = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    nucleus_3d_image = np.moveaxis(nucleus_3d_image, 0, -1)
    return nucleus_3d_image


def get_nucleus_xsections(nucleus_3d_image, output_folder):
    x_start, x_end, step = 1850, 2360, 1
    for x_slice in range(x_start, x_end, step):
        xsection = nucleus_3d_image[680: 1020, x_slice, :]

        img_path = os.path.join(output_folder, "xsection_nucleus_" + str(x_slice) + ".png")
        cv2.imwrite(img_path, cv2.resize(xsection, (4*xsection.shape[1], 4*xsection.shape[0])))


if __name__ == "__main__":
    input_folder = 'temp/czi_layers'
    output_folder = 'temp/nucleus_layers'

    nucleus_3d_img = get_3d_image(input_folder, 'nucleus')
    get_nucleus_xsections(nucleus_3d_img, output_folder)
