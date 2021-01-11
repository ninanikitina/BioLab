import os

from czifile import CziFile
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np


# Typical shape is (1, 1, 2, 1, 38, 6355, 6359, 1)

def get_normalization_th(img, noise_th):
    hist = np.squeeze(cv2.calcHist([img], [0], None, [65536], [0, 65536]))

    norm_th, sum = 1, 0
    while (sum / img.size < noise_th):
        sum += hist[-norm_th]
        norm_th += 1

    return norm_th


def normalization(img, norm_th):
    img[np.where(img > 65535 - norm_th)] = 65535 - norm_th
    img = cv2.normalize(img, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    img = (img / 256).astype(np.uint8)

    return img


def get_normalized_img_path(img_name, layer):
    if not os.path.exists('temp/czi_layers'):
        os.makedirs('temp/czi_layers')

    img_path_norm = os.path.join("temp/czi_layers", img_name + '_layer_' + str(layer) + '.png')
    return img_path_norm


def run_czi_reader(img_path, czi_img_channel, noise_th):
    img_name = os.path.splitext(os.path.basename(img_path))[0]

    with CziFile(img_path) as czi:
        image_arrays = czi.asarray()

    middle_layer = image_arrays.shape[4] // 2
    norm_th = get_normalization_th(image_arrays[0, 0, czi_img_channel, 0, middle_layer, :, :, 0], noise_th)

    for i in tqdm(range(image_arrays.shape[4])):
        img_path_norm = get_normalized_img_path(img_name, i)

        norm_image = normalization(image_arrays[0, 0, czi_img_channel, 0, i, :, :, 0], norm_th)

        padded_image = cv2.copyMakeBorder(norm_image, 256, 256, 256, 256, cv2.BORDER_CONSTANT, None, 0)
        cv2.imwrite(img_path_norm, padded_image)

    print("Czi image has {} layers of shape (h={}, w={})\n".format(image_arrays.shape[4],
                                                                   image_arrays.shape[5] + 512,
                                                                   image_arrays.shape[6] + 512))
    return image_arrays.shape[5] + 512, image_arrays.shape[6] + 512, image_arrays.shape[4]


if __name__ == '__main__':
    img_path = r"D:\BioLab\img\3Dimg_for_tests\3D\20x\control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003.czi"
    czi_img_channel = 1
    noise_th = 0.0001

    h, w, layers = run_czi_reader(img_path, czi_img_channel, noise_th)
