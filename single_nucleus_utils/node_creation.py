import os
import glob
import pickle
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from single_nucleus_utils.find_actin_fiber import ActinFiber
from single_nucleus_utils.analyze_acting_masks_contour_xsection_multiple import plot_actin_fibers


class Node:
    def __init__(self, x, y, z, cnt, actin_id, actin_side):
        self.x = x
        self.y = y
        self.z = z
        self.n = 1
        self.cnt = cnt
        self.actin_id = actin_id
        self.actin_side = actin_side


def plot_nodes(actin_fibers, nodes):
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

    xdata, ydata, zdata = [], [], []
    for node in nodes:
        xdata.append(node.x)
        ydata.append(node.y)
        zdata.append(node.z)
    color_x = 1.0
    color_y = 0
    color_z = 0
    if xdata:
        ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]] * len(xdata), s=50, cmap='Greens')
    plt.show()


if __name__ == "__main__":
    file_to_read = open(r"D:\BioLab\src\single_nucleus_utils\actin_data_long.obj", "rb")
    actin_fibers = pickle.load(file_to_read)
    actin_fibers = [actin for actin in actin_fibers if actin.n > 30]

    nodes = []
    for actin_id, actin in enumerate(actin_fibers):
        nodes.append(Node(actin.xs[0], actin.ys[0], actin.zs[0], actin.cnts[0], actin_id, 'left'))
        nodes.append(Node(actin.xs[-1], actin.ys[-1], actin.zs[-1], actin.cnts[-1], actin_id, 'right'))

    for left_node in nodes[::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if left_node.x - 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(left_node.x - 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [left_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers.pop(actin_id_to_break)
            break_index = actin_to_break.xs.index(left_node.x - 1)

            new_left_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_left_actin.xs = actin_to_break.xs[:break_index + 1]
            new_left_actin.ys = actin_to_break.ys[:break_index + 1]
            new_left_actin.zs = actin_to_break.zs[:break_index + 1]
            new_left_actin.last_layer = actin_to_break.last_layer[:break_index + 1]
            new_left_actin.cnts = actin_to_break.cnts[:break_index + 1]
            new_left_actin.n = len(new_left_actin.xs)

            new_right_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)

            actin_fibers.append(new_left_actin)
            actin_fibers.append(new_right_actin)

    # -------------------------------------------------------- #
    for right_node in nodes[1::2]:
        actin_id_to_break = None

        actins_to_check = [[actin_id, actin] for actin_id, actin in enumerate(actin_fibers)
                           if right_node.x + 1 in actin.xs]
        cnts_to_check = [actin.cnts[actin.xs.index(right_node.x + 1)] for _, actin in actins_to_check]

        node_mask = np.zeros((1300, 1300), dtype=np.uint8)
        cv2.drawContours(node_mask, [right_node.cnt], -1, 255, -1)
        for cnt_id, cnt in enumerate(cnts_to_check):
            cnt_mask = np.zeros((1300, 1300), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)

            if np.any(cv2.bitwise_and(node_mask, cnt_mask)):
                actin_id_to_break = actins_to_check[cnt_id][0]

        if actin_id_to_break is not None:
            actin_to_break = actin_fibers.pop(actin_id_to_break)
            break_index = actin_to_break.xs.index(right_node.x + 1)

            new_left_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_left_actin.xs = actin_to_break.xs[:break_index + 1]
            new_left_actin.ys = actin_to_break.ys[:break_index + 1]
            new_left_actin.zs = actin_to_break.zs[:break_index + 1]
            new_left_actin.last_layer = actin_to_break.last_layer[:break_index + 1]
            new_left_actin.cnts = actin_to_break.cnts[:break_index + 1]
            new_left_actin.n = len(new_left_actin.xs)

            new_right_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)

            actin_fibers.append(new_left_actin)
            actin_fibers.append(new_right_actin)

    actin_fibers = [actin for actin in actin_fibers if actin.n > 5]
    plot_nodes(actin_fibers, nodes)
    # plot_actin_fibers(actin_fibers)
