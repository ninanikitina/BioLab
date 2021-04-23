import os
import glob
import pickle
from collections import defaultdict
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from single_nucleus_utils.find_actin_fiber import ActinFiber
from single_nucleus_utils.analyze_acting_masks_contour_xsection_multiple import plot_actin_fibers

H = 150
L = 400

class Node:
    def __init__(self, x, y, z, cnt, actin_id):
        self.x = x
        self.y = y
        self.z = z
        self.n = 1
        self.cnt = cnt
        self.actin_ids = [actin_id]


def is_point_in_triangle(x, y, x_test, y_test):
    is_x_in_triangle = x_test < x + H
    is_y_in_triangle = -L*(x_test - x)/(2*H) + y < y_test < L*(x_test - x)/(2*H) + y

    return is_x_in_triangle and is_y_in_triangle


def plot_nodes(actin_fibers, nodes, pairs):
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

    for right_id, left_id in pairs:
        right_node, left_node = nodes[right_id], nodes[left_id]
        plt.plot((right_node.x, left_node.x), (right_node.y, left_node.y), (right_node.z, left_node.z), color='black')

    plt.show()


if __name__ == "__main__":
    file_to_read = open(r"D:\BioLab\src\single_nucleus_utils\actin_data_long.obj", "rb")
    actin_fibers = pickle.load(file_to_read)
    actin_fibers = [actin for actin in actin_fibers if actin.n > 30]

    nodes = []
    for actin_id, actin in enumerate(actin_fibers):
        nodes.append(Node(actin.xs[0], actin.ys[0], actin.zs[0], actin.cnts[0], actin_id))          # left side
        nodes.append(Node(actin.xs[-1], actin.ys[-1], actin.zs[-1], actin.cnts[-1], actin_id))      # right side
        actin.left_node_id = len(nodes) - 2
        actin.right_node_id = len(nodes) - 1

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
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(left_node.x - 1)

            new_right_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(left_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(left_node)

            # update node
            left_node.actin_ids.append(actin_id_to_break)           # adding old actin which attached from left
            left_node.actin_ids.append(len(actin_fibers) - 1)       # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop( old_right_node.actin_ids.index(actin_id_to_break) )
            old_right_node.actin_ids.append(len(actin_fibers) - 1)

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
            actin_to_break = actin_fibers[actin_id_to_break]
            break_index = actin_to_break.xs.index(right_node.x + 1)

            new_right_actin = ActinFiber(-1, -1, -1, -1, -1)
            new_right_actin.xs = actin_to_break.xs[break_index + 1:]
            new_right_actin.ys = actin_to_break.ys[break_index + 1:]
            new_right_actin.zs = actin_to_break.zs[break_index + 1:]
            new_right_actin.last_layer = actin_to_break.last_layer[break_index + 1:]
            new_right_actin.cnts = actin_to_break.cnts[break_index + 1:]
            new_right_actin.n = len(new_right_actin.xs)
            new_right_actin.left_node_id = nodes.index(right_node)
            new_right_actin.right_node_id = actin_to_break.right_node_id

            actin_to_break.xs = actin_to_break.xs[:break_index + 1]
            actin_to_break.ys = actin_to_break.ys[:break_index + 1]
            actin_to_break.zs = actin_to_break.zs[:break_index + 1]
            actin_to_break.last_layer = actin_to_break.last_layer[:break_index + 1]
            actin_to_break.cnts = actin_to_break.cnts[:break_index + 1]
            actin_to_break.n = len(actin_to_break.xs)

            actin_fibers.append(new_right_actin)

            # update old actin
            actin_to_break.right_node_id = nodes.index(right_node)

            # update node
            right_node.actin_ids.append(actin_id_to_break)      # adding old actin which attached from left
            right_node.actin_ids.append(len(actin_fibers) - 1)  # adding new actin which attached from right

            # update old right node
            old_right_node = nodes[new_right_actin.right_node_id]
            old_right_node.actin_ids.pop( old_right_node.actin_ids.index(actin_id_to_break) )
            old_right_node.actin_ids.append(len(actin_fibers) - 1)

    # Actin merging code (triangle approach), create merge candidate dictionary
    right_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                   if len(node.actin_ids) == 1 and actin_fibers[node.actin_ids[0]].right_node_id == node_id]

    left_nodes = [(node_id, node) for node_id, node in enumerate(nodes)
                   if len(node.actin_ids) == 1 and actin_fibers[node.actin_ids[0]].left_node_id == node_id]

    node_to_candidates = defaultdict(lambda: [])
    for righ_node_id, right_node in right_nodes:
        for left_node_id, left_node in left_nodes:
            if is_point_in_triangle(right_node.x, right_node.y, left_node.x, left_node.y):
                node_to_candidates[righ_node_id].append(
                    [left_node_id, np.sqrt((left_node.x - right_node.x)**2 + (left_node.y - right_node.y)**2)])

    node_to_candidates_list = []
    for k, v in node_to_candidates.items():
        node_to_candidates[k] = sorted(v, key=lambda x: x[1])
        node_to_candidates_list.append([k, v])

    # Create actual pairs of nodes to connect
    pairs = []
    while len(node_to_candidates_list) > 0:
        # find min distance
        min_distance = 10000
        r, l, pop_idx = None, None, None
        for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
            if min_distance > candidate_distance_list[0][1]:
                min_distance = candidate_distance_list[0][1]
                r, l, pop_idx = right_node, candidate_distance_list[0][0], idx
        pairs.append([r, l])
        node_to_candidates_list.pop(pop_idx)

        # remove left node from others candidates
        for right_node, candidate_distance_list in node_to_candidates_list:
            lefts = [item[0] for item in candidate_distance_list]
            if l in lefts:
                candidate_distance_list.pop(lefts.index(l))

        indicies_to_remove = []
        for idx, (right_node, candidate_distance_list) in enumerate(node_to_candidates_list):
            if len(candidate_distance_list) == 0:
                indicies_to_remove.append(idx)

        for idx in indicies_to_remove[::-1]:
            node_to_candidates_list.pop(idx)


    # actin_fibers = [actin for actin in actin_fibers if actin.n > 5]
    plot_nodes(actin_fibers, nodes, pairs)
    # plot_actin_fibers(actin_fibers)
