import os

from czifile import CziFile
from tqdm import tqdm
import cv2.cv2 as cv2

from multiple_nuclei_utils.czi_reader import get_normalization_th, normalization, get_normalized_img_path


def run_czi_reader_for_nucleus(img_name, image_arrays, czi_img_channel):

    for i in tqdm(range(image_arrays.shape[4])):
        img_path = get_normalized_img_path(img_name + "_nucleus", i)

        image = image_arrays[0, 0, czi_img_channel, 0, i, :, :, 0]

        cv2.imwrite(img_path, image)

    print("Czi image has {} layers of shape (h={}, w={})\n".format(image_arrays.shape[4],
                                                                   image_arrays.shape[5],
                                                                   image_arrays.shape[6]))


def run_czi_reader_for_actin(img_name, image_arrays, czi_img_channel):
    for i in tqdm(range(image_arrays.shape[4])):
        img_path = get_normalized_img_path(img_name + "_actin", i)

        image = image_arrays[0, 0, czi_img_channel, 0, i, :, :, 0]

        cv2.imwrite(img_path, image)

    print("Czi image has {} layers of shape (h={}, w={})\n".format(image_arrays.shape[4],
                                                                   image_arrays.shape[5],
                                                                   image_arrays.shape[6]))

def run_czi_reader_255(CONFOCAL_IMG, NUCLEI_CHANNEL, ACTIN_CHANNEL):
    img_name = os.path.splitext(os.path.basename(CONFOCAL_IMG))[0]
    with CziFile(CONFOCAL_IMG) as czi:
        image_arrays = czi.asarray()

    run_czi_reader_for_nucleus(img_name, image_arrays, NUCLEI_CHANNEL)
    run_czi_reader_for_actin(img_name, image_arrays, ACTIN_CHANNEL)
    return image_arrays.shape[5], image_arrays.shape[6], image_arrays.shape[4]

if __name__ == '__main__':
    img_path = r"D:\BioLab\img\Buer_img\Test1-L8-0.12.czi"
    czi_actin_channel = 0
    czi_nucleus_channel = 2

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    with CziFile(img_path) as czi:
        image_arrays = czi.asarray()

    run_czi_reader_for_nucleus(img_name, image_arrays, czi_nucleus_channel)
    run_czi_reader_for_actin(img_name, image_arrays, czi_actin_channel)
