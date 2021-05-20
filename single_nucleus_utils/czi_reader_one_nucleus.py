import os

from czifile import CziFile
from tqdm import tqdm
import cv2.cv2 as cv2

from multiple_nuclei_utils.czi_reader import get_normalization_th, normalization, get_normalized_img_path


def run_czi_reader_for_nucleus(img_name, image_arrays, czi_img_channel, noise_th):
    middle_layer = image_arrays.shape[4] // 2
    norm_th = get_normalization_th(image_arrays[0, 0, czi_img_channel, 0, middle_layer, :, :, 0], noise_th)

    for i in tqdm(range(image_arrays.shape[4])):
        img_path_norm = get_normalized_img_path(img_name + "_nucleus", i)

        norm_image = normalization(image_arrays[0, 0, czi_img_channel, 0, i, :, :, 0], norm_th)

        cv2.imwrite(img_path_norm, norm_image)

    print("Czi image has {} layers of shape (h={}, w={})\n".format(image_arrays.shape[4],
                                                                   image_arrays.shape[5],
                                                                   image_arrays.shape[6]))
    return image_arrays.shape[5], image_arrays.shape[6], image_arrays.shape[4], norm_th


def run_czi_reader_for_actin(img_name, image_arrays, czi_img_channel, norm_th):
    for i in tqdm(range(image_arrays.shape[4])):
        img_path_norm = get_normalized_img_path(img_name + "_actin", i)

        norm_image = normalization(image_arrays[0, 0, czi_img_channel, 0, i, :, :, 0], norm_th)

        cv2.imwrite(img_path_norm, norm_image)

    print("Czi image has {} layers of shape (h={}, w={})\n".format(image_arrays.shape[4],
                                                                   image_arrays.shape[5],
                                                                   image_arrays.shape[6]))
    return image_arrays.shape[5], image_arrays.shape[6], image_arrays.shape[4]


def run_czi_reader(CONFOCAL_IMG, NUCLEI_CHANNEL, ACTIN_CHANNEL, NOISE_TH):
    img_name = os.path.splitext(os.path.basename(CONFOCAL_IMG))[0]
    with CziFile(CONFOCAL_IMG) as czi:
        image_arrays = czi.asarray()

    _, _, _, norm_th = run_czi_reader_for_nucleus(img_name, image_arrays, NUCLEI_CHANNEL, NOISE_TH)
    _, _, _ = run_czi_reader_for_actin(img_name, image_arrays, ACTIN_CHANNEL, norm_th)
    return image_arrays.shape[5], image_arrays.shape[6], image_arrays.shape[4]

if __name__ == '__main__':
    img_path = r"D:\BioLab\img\3Dimg_for_tests\3D\63x\control siRNA 63x_3x1_5_um_stack_11_16_Airyscan_Processing_Stitch.czi"
    czi_actin_channel = 0
    czi_nucleus_channel = 1
    noise_th = 0.0001

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    with CziFile(img_path) as czi:
        image_arrays = czi.asarray()

    print(image_arrays.shape)
    print(image_arrays[0, 0, 0, 0, :, :, 0].dtype == "uint16")
    _, _, _, norm_th = run_czi_reader_for_nucleus(img_name, image_arrays, czi_nucleus_channel, noise_th)
    _, _, _ = run_czi_reader_for_actin(img_name, image_arrays, czi_actin_channel, norm_th)
