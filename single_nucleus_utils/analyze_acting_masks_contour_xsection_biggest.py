import os
import glob
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import pickle

from multiple_nuclei_utils.cut_nuclei import get_cnt_center
from single_nucleus_utils.find_actin_fiber import ActinFiber

MIN_FIBER_LENGTH_FINAL = 20 #5, 20, 40
GAP = 5 #5, 15, 100 px

input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V1-with_weight_correction_475"
#input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V2-with_weight_correction_20_e200"
#input_folder = r"D:\BioLab\img\Buer_big_nucleous_training_img_and_mask\layers_unet_mask_padding"
output_folder = r"C:\Users\nnina\Desktop\test2"  # r"D:\BioLab\Tests_for_hand_labling\Final_masks"


class NewLayerContours(object):
    def __init__(self, x, y, z, cnt):
        self.x = x
        self.y = y
        self.z = z
        self.cnt = cnt
        self.xsection = 0
        self.parent = None


def draw_actin_pairs(actin_pairs):
    ax = plt.axes(projection='3d')
    for actin_1, actin_2 in actin_pairs:
        xdata = actin_1.xs + actin_2.xs
        ydata = actin_1.ys + actin_2.ys
        zdata = actin_1.zs + actin_2.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]]*len(xdata), cmap='Greens')
    plt.show()


def get_3d_img():
    object_layers = []

    for img_path in glob.glob(input_folder + r"\*"):
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        layer = int(img_name.rsplit("_", 1)[1])

        img = cv2.imread(img_path, 0)
        img = np.flip(img, axis=0)
        object_layers.append([img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    img_3d = np.asarray([mask for mask, layer in object_layers], dtype=np.uint8)

    return img_3d


def get_all_actin_coords_and_cnts(xsection, x):
    cnts = cv2.findContours(xsection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    layer_data = []
    for cnt in cnts:
        z, y = get_cnt_center(cnt)
        layer_data.append(NewLayerContours(x, y, z, cnt))

    return layer_data


def get_actin_fibers(img_3d):
    actin_fibers = []

    for x_slice in range(img_3d.shape[0]):
        print(x_slice)

        xsection = img_3d[x_slice, :, :]

        layer_data = get_all_actin_coords_and_cnts(xsection, x_slice)

        if x_slice == 0 or not actin_fibers:
            actin_fibers.extend([ActinFiber(new_layer_cnt.x,
                                            new_layer_cnt.y,
                                            new_layer_cnt.z,
                                            x_slice,
                                            new_layer_cnt.cnt)
                                 for new_layer_cnt in layer_data])
        else:
            actin_fibers_from_previous_layer = [item for item in actin_fibers if item.last_layer[-1] == x_slice - 1]
            print("Number of actins from previous layer: {}".format(len(actin_fibers_from_previous_layer)))

            # find parents for all contours on new layer
            for new_layer_cnt in layer_data:
                for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                    new_layer_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(new_layer_cnt_mask, [new_layer_cnt.cnt], -1, 255, -1)

                    actin_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(actin_cnt_mask, [actin_fiber.cnts[-1]], -1, 255, -1)

                    intersection = np.count_nonzero(cv2.bitwise_and(new_layer_cnt_mask, actin_cnt_mask))
                    if intersection > 0 and intersection > new_layer_cnt.xsection:
                        new_layer_cnt.xsection = intersection
                        new_layer_cnt.parent = i

            # assign contour to actin fibers from previous layer
            for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                children_cnts = [new_layer_cnt for new_layer_cnt in layer_data if new_layer_cnt.parent == i]

                if len(children_cnts) == 1:
                    new_layer_cnt = children_cnts[0]
                    actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)

                if len(children_cnts) > 1:
                    max_intersection, idx = 0, -1
                    for i, child_cnt in enumerate(children_cnts):
                        if child_cnt.xsection > max_intersection:
                            max_intersection = child_cnt.xsection
                            idx = i

                    new_layer_cnt = children_cnts[idx]
                    actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)

                    for i, child_cnt in enumerate(children_cnts):
                        if i != idx:
                            actin_fibers.append(ActinFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))

            for child_cnt in layer_data:
                if child_cnt.parent is None:
                    actin_fibers.append(ActinFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))


    return actin_fibers


def plot_actin_fibers(actin_fibers):
    ax = plt.axes(projection='3d')
    for fiber in actin_fibers:
        if len(np.unique(fiber.zs)) < 1:
            continue

        # Draw only center points
        xdata = fiber.xs
        ydata = fiber.ys
        zdata = fiber.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), cmap='Greens')
    plt.show()


if __name__ == "__main__":
    img_3d = get_3d_img()
    actin_fibers = get_actin_fibers(img_3d)

    with open("actin_data_long2.obj", "wb") as file_to_save:   # change back to "actin-data_long.obj"
        pickle.dump(actin_fibers, file_to_save)

    actin_fibers = [actin for actin in actin_fibers if actin.n > MIN_FIBER_LENGTH_FINAL]
    plot_actin_fibers(actin_fibers)
