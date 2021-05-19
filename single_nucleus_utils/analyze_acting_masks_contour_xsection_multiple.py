import os
import glob
import pickle
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from multiple_nuclei_utils.cut_nuclei import get_cnt_center
from single_nucleus_utils.find_actin_fiber import ActinFiber


MIN_FIBER_LENGTH_FINAL = 1 #5, 20, 40
GAP = 15 #5, 15, 100 px

input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V1-with_weight_correction_475"
#input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V1-with_weight_correction_10_200"

output_folder = r"C:\Users\nnina\Desktop\test"  # r"D:\BioLab\Tests_for_hand_labling\Final_masks"


class NewLayerContours(object):
    def __init__(self, x, y, z, cnt):
        self.x = x
        self.y = y
        self.z = z
        self.cnt = cnt
        self.processed = False
        self.parents = []


class ShortActinFiber(ActinFiber):
    def __init__(self, id, x, y, z, layer, cnt):
        ActinFiber.__init__(self, x, y, z, layer, cnt)
        self.id = id
        self.child_ids = []
        self.cnt_children = []

    def add_child(self, child_id):
        self.child_ids.append(child_id)


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
            actin_fibers.extend([ShortActinFiber(actin_number,
                                                 new_layer_cnt.x,
                                                 new_layer_cnt.y,
                                                 new_layer_cnt.z,
                                                 x_slice,
                                                 new_layer_cnt.cnt)
                                 for actin_number, new_layer_cnt in enumerate(layer_data)])
        else:
            actin_fibers_from_previous_layer = [item for item in actin_fibers if item.last_layer[-1] == x_slice - 1]
            print("Number of actins from previous layer: {}".format(len(actin_fibers_from_previous_layer)))

            # find parents for all contours on new layer
            for new_layer_cnt in layer_data:
                for actin_fiber in actin_fibers_from_previous_layer:
                    new_layer_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(new_layer_cnt_mask, [new_layer_cnt.cnt], -1, 255, -1)

                    actin_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(actin_cnt_mask, [actin_fiber.cnts[-1]], -1, 255, -1)

                    intersection = np.count_nonzero(cv2.bitwise_and(new_layer_cnt_mask, actin_cnt_mask))
                    if intersection > 0:
                        new_layer_cnt.parents.append(actin_fiber.id)

            # find all contour children for each actin
            for actin in actin_fibers_from_previous_layer:
                for i, new_layer_cnt in enumerate(layer_data):
                    if actin.id in new_layer_cnt.parents:
                        actin.cnt_children.append(i)

            # create new actins and update actin children
            for i, new_layer_cnt in enumerate(layer_data):
                if new_layer_cnt.processed:
                    continue

                if len(new_layer_cnt.parents) == 1 and len(actin_fibers[new_layer_cnt.parents[0]].cnt_children) == 1:
                    actin_fibers[new_layer_cnt.parents[0]].update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)
                    # actin_fibers[new_layer_cnt.parents[0]].cnt_children = []
                    new_layer_cnt.processed = True

                elif len(new_layer_cnt.parents) == 1 and len(actin_fibers[new_layer_cnt.parents[0]].cnt_children) > 1:
                    for cnt_id in actin_fibers[new_layer_cnt.parents[0]].cnt_children:
                        child_cnt = layer_data[cnt_id]
                        actin_fibers.append(ShortActinFiber(len(actin_fibers), child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))
                        actin_fibers[new_layer_cnt.parents[0]].add_child(len(actin_fibers) - 1)

                        new_layer_cnt.processed = True

                elif len(new_layer_cnt.parents) > 1:
                    actin_fibers.append(ShortActinFiber(len(actin_fibers), new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt))
                    for parent_id in new_layer_cnt.parents:
                        actin_fibers[parent_id].add_child(len(actin_fibers) - 1)
                    new_layer_cnt.processed = True

                else:
                    actin_fibers.append(ShortActinFiber(len(actin_fibers), new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt))
                    new_layer_cnt.processed = True

            for actin in actin_fibers_from_previous_layer:
                actin.cnt_children = []


            # # assign contour to actin fibers from previous layer
            # for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
            #     children_cnts = [new_layer_cnt for new_layer_cnt in layer_data if i in new_layer_cnt.parents]
            #
            #     if len(children_cnts) == 1 and len(children_cnts[0].parents) == 1:
            #         new_layer_cnt = children_cnts[0]
            #         actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)
            #         new_layer_cnt.processed = True
            #
            #     elif len(children_cnts) == 1 and len(children_cnts[0].parents) > 1:
            #         child_cnt = children_cnts[0]
            #
            #         if not child_cnt.processed:
            #             actin_fibers.append(ShortActinFiber(len(actin_fibers), child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))
            #
            #         actin_fiber.add_child(actin_fibers[-1].id)
            #         child_cnt.processed = True
            #
            #     if len(children_cnts) > 1:
            #         # max_intersection, idx = 0, -1
            #         # for i, child_cnt in enumerate(children_cnts):
            #         #     if child_cnt.xsection > max_intersection:
            #         #         max_intersection = child_cnt.xsection
            #         #         idx = i
            #
            #         # new_layer_cnt = children_cnts[idx]
            #         # actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)
            #
            #         for i, child_cnt in enumerate(children_cnts):
            #             actin_fibers.append(ShortActinFiber(len(actin_fibers), child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))
            #             actin_fiber.add_child(actin_fibers[-1].id)
            #
            # for child_cnt in layer_data:
            #     if child_cnt.parent is None:
            #         actin_fibers.append(ShortActinFiber(len(actin_fibers), child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))

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

    with open("actin_data.obj", "wb") as file_to_save:
        pickle.dump(actin_fibers, file_to_save)


    actin_fibers = [actin for actin in actin_fibers if actin.n > MIN_FIBER_LENGTH_FINAL]

    plot_actin_fibers(actin_fibers)
