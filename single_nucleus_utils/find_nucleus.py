import os
import glob
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

from multiple_nuclei_utils.cut_nuclei import get_cnts, draw_cnts, get_cnt_center


def get_nucleus_cnt(img):
    cnts = get_cnts((img))
    cnt = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]
    return cnt


def find_biggest_nucleus_layer(input_folder):
    biggest_layer = 0
    nucleus_area = 0
    biggest_nucleus_mask = None
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_nucleus_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        current_nucleus_cnt = get_nucleus_cnt(nucleus_img)
        current_nucleus_cnt_area = cv2.contourArea(current_nucleus_cnt)
        if current_nucleus_cnt_area > nucleus_area:
            nucleus_area = current_nucleus_cnt_area
            biggest_layer = layer
            biggest_nucleus_mask = draw_cnts(nucleus_img.shape[:2], [current_nucleus_cnt])

        # cnt_mask = draw_cnts(nucleus_img.shape[:2], [current_nucleus_cnt])
        # cv2.imwrite(img_path.rsplit(".", 1)[0] + "_mask.png", cnt_mask)

    return biggest_layer, biggest_nucleus_mask


def get_actin_3d_image(input_folder, output_folder, biggest_layer, biggest_nucleus_mask):
    actin_cap_layers = []

    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_actin_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        if layer >= biggest_layer:
            actin_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            actin_cap_layer = cv2.bitwise_and(actin_img, biggest_nucleus_mask)
            actin_cap_layers.append([actin_cap_layer, layer])

            actin_cap_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(actin_cap_path, actin_cap_layer)

    actin_cap_layers = sorted(actin_cap_layers, key=lambda x: x[1], reverse=True)
    actin_3d_image = np.asarray([img for img, layer in actin_cap_layers], dtype=np.uint8)

    return actin_3d_image


def get_all_actin_coords(xsection, i):
    _, mask = cv2.threshold(xsection, 50, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    centers = []
    for cnt in cnts:
        x, y = get_cnt_center(cnt)
        centers.append([x-680, y/2, i - 1980])

    return centers


def get_actin_xsections(actin_3d_img, output_folder):
    actin_coords = []
    for i in range(1980, 2350, 1):
        xsection = actin_3d_img[:, 680: 1020, i]
        # xsection = cv2.resize(xsection, (340*4, 25*4), interpolation=cv2.INTER_CUBIC)
        actin_coords.extend(get_all_actin_coords(xsection, i))
        # cv2.imwrite(os.path.join(output_folder, "xsection_" + str(i) + ".png"), xsection)

    ax = plt.axes(projection='3d')
    xdata = [z for x, y, z in actin_coords]
    ydata = [x for x, y, z in actin_coords]
    zdata = [y for x, y, z in actin_coords]
    ax.scatter3D(xdata, ydata, zdata, cmap='Greens')
    # ax.set_xlim3d(0, 2350 - 1980)
    # ax.set_ylim3d(0, 300)
    # ax.set_zlim3d(0, 25)
    plt.show()


if __name__ == "__main__":
    input_folder = 'temp/czi_layers'
    output_folder = 'temp/actin_and_nucleus_layers'

    biggest_layer, biggest_nucleus_mask = find_biggest_nucleus_layer(input_folder)
    actin_3d_img = get_actin_3d_image(input_folder, output_folder, biggest_layer, biggest_nucleus_mask)
    get_actin_xsections(actin_3d_img, output_folder)
