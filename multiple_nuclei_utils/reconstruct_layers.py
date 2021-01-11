import os
import glob

import numpy as np
import cv2.cv2 as cv2
from tqdm import tqdm

from multiple_nuclei_utils.cut_nuclei import get_cnt_center


def get_mask_coordinates_from_img_name(img_name):
    str_coords = img_name.rsplit('.', 1)[0].rsplit('_', 2)[1:]
    return int(str_coords[0]), int(str_coords[1])


def get_primary_cnt(cnts):
    cnt_centers = [get_cnt_center(cnt) for cnt in cnts if len(cnt) > 100]

    if len(cnt_centers) == 0:
        return None

    if len(cnt_centers) == 1:
        cnt = cnts[0]
    else:
        idx, dist = 0, np.inf
        for i, (x, y) in enumerate(cnt_centers):
            if (x - 256) ** 2 + (y - 256) ** 2 < dist:
                idx, dist = i, (x - 256) ** 2 + (y - 256) ** 2
        cnt = cnts[idx]

    return cnt


def get_layer_imgs(folder_path, layer):
    imgs = glob.glob(os.path.join(folder_path, "*_layer_" + str(layer) + "_*.png"))
    return imgs


def merge_individual_masks(folder_path, output_folder_path, layers, h, w):
    if folder_path is None:
        if not os.path.exists('temp/true_nucleus_masks'):
            raise RuntimeError("There is no folder {}\nCan't process images".format("temp/true_nucleus_masks"))

        if not os.path.exists('temp/reconstructed_layers'):
            os.makedirs('temp/reconstructed_layers')

        folder_path = 'temp/true_nucleus_masks'
        output_folder_path = 'temp/reconstructed_layers'

    for layer in tqdm(range(layers)):
        layer_imgs = get_layer_imgs(folder_path, layer)

        merged_mask = np.zeros((h, w), dtype=np.uint8)
        for img_path in layer_imgs:
            x_coord, y_coord = get_mask_coordinates_from_img_name(img_path)

            img = cv2.imread(img_path)[:, :, 0]
            cnts = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
            cnt = get_primary_cnt(cnts)

            if cnt is not None:
                cnt_mask = np.zeros(img.shape, dtype=np.uint8)
                cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

                merged_mask[y_coord - 256: y_coord + 256, x_coord - 256: x_coord + 256] += cnt_mask

        if layer_imgs:
            img_name = os.path.basename(layer_imgs[0]).rsplit('_', 2)[0]
            img_path_to_save = os.path.join(output_folder_path, img_name + ".png")
            cv2.imwrite(img_path_to_save, merged_mask)


if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\unet_prediction\control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003\Prediction_7_control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003"
    output_folder_path = r"D:\BioLab\img\big_masks\control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003\Big_mask_5"
    layers, h, w = 31, 8438, 8444

    merge_individual_masks(folder_path, output_folder_path, layers, h, w)
