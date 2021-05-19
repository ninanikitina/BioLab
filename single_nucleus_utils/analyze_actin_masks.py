import os
import glob
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import csv

from multiple_nuclei_utils.cut_nuclei import get_cnt_center
from single_nucleus_utils.find_actin_fiber import ActinFiber

MIN_FIBER_LENGTH = 1 #5, 20, 40
MIN_FIBER_LENGTH_FINAL = 1 #5, 20, 40
Y_MAX_SHIFT = 16 #4, 8, 16 px
Z_MAX_SHIFT = 8 #2, 4, 8 px
GAP = 15 #5, 15, 100 px
PAIR_MIN_DISTANCE = 10 #20
PAIR_MIN_TG = 0.2 #0.2


#input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V1-with_weight_correction_475"
#input_folder = r"D:\BioLab\img\Actin__training_img_and_masks\Unet_results\Unet_prediction_V1-with_weight_correction_10_200"
input_folder = r"D:\BioLab\img\Buer_big_nucleous_training_img_and_mask\layers_unet_mask_padding"

output_folder = r"C:\Users\nnina\Desktop\test" #r"D:\BioLab\Tests_for_hand_labling\Final_masks"


def analyze_actin_tracks(actin_fibers):
    alphas, offsets = [], []
    for actin in actin_fibers:
        vx, vy, cx, cy = actin.line

        alphas.append(vy / vx)
        offsets.append(cy - cx * (vy / vx))

    plt.scatter(offsets, alphas)
    plt.show()

    return alphas, offsets


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


def save_masks_and_get_3d_img():
    object_layers = []

    for img_path in glob.glob(input_folder + r"\*"):
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        layer = int(img_name.rsplit("_", 1)[1])

        img = cv2.imread(img_path, 0)
        img = np.flip(img, axis=0)
        _, mask = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
        object_layers.append([mask, layer])

        mask_path = os.path.join(output_folder, img_name + "_mask" + img_ext)
        cv2.imwrite(mask_path, mask)

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    img_3d = np.asarray([mask for mask, layer in object_layers], dtype=np.uint8)

    return img_3d


def get_actin_stat(actin_fibers):
    header_row = ["#", "Actin Length", "Actin Xsection", "Actin Volume", "Number of fiber layers", "Max gap"]
    with open('actin_stat.csv', mode='w') as stat_file:
        csv_writer = csv.writer(stat_file, delimiter=',')

        csv_writer.writerow(header_row)

        i = 1
        for fiber in actin_fibers:
            if fiber.n > MIN_FIBER_LENGTH:
                csv_writer.writerow([str(i)] + fiber.get_stat())
                i += 1


def get_all_actin_coords(xsection, x):
    cnts = cv2.findContours(xsection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    centers = []
    for cnt in cnts:
        z, y = get_cnt_center(cnt)
        centers.append([x, y, z, cnt])

    return centers


def get_3d_plot(img_3d):
    actin_fibers = []
    actin_coords = []

    for x_slice in range(img_3d.shape[0]):
        xsection = img_3d[x_slice, :, :]

        current_layer_coords = get_all_actin_coords(xsection, x_slice)

        if x_slice == 0 or not actin_fibers:
            actin_fibers.extend([ActinFiber(x, y, z, x_slice, cnt) for x, y, z, cnt in current_layer_coords])
        else:
            for x, y, z, cnt in current_layer_coords:
                was_added = False
                for actin_fiber in actin_fibers:
                    if actin_fiber.ys[-1] - Y_MAX_SHIFT < y < actin_fiber.ys[-1] + Y_MAX_SHIFT and \
                            actin_fiber.zs[-1] - Z_MAX_SHIFT < z < actin_fiber.zs[-1] + Z_MAX_SHIFT and \
                            x - actin_fiber.last_layer[-1] < GAP:
                        actin_fiber.update(x, y, z, x_slice, cnt)
                        was_added = True
                        break

                if not was_added:
                    actin_fibers.append(ActinFiber(x, y, z, x_slice, cnt))

        actin_coords.extend(current_layer_coords)
        # cv2.imwrite(os.path.join(output_folder, "xsection_" + str(i) + ".png"), xsection)

    actin_fibers = [actin for actin in actin_fibers if actin.n > MIN_FIBER_LENGTH]

    for actin in actin_fibers:
        actin.fit_line()

    alphas, offsets = analyze_actin_tracks(actin_fibers)

    actin_pairs = []
    for i in range(len(actin_fibers)):
        for j in range(i + 1, len(actin_fibers), 1):
            if abs(alphas[i] - alphas[j]) < PAIR_MIN_TG and abs(offsets[i] - offsets[j]) < PAIR_MIN_DISTANCE:
                actin_pairs.append([actin_fibers[i], actin_fibers[j]])

    draw_actin_pairs(actin_pairs)
    for i in range(len(actin_fibers)):
        if actin_fibers[i].merged:
            continue

        for j in range(i + 1, len(actin_fibers), 1):
            if abs(alphas[i] - alphas[j]) < PAIR_MIN_TG and abs(offsets[i] - offsets[j]) < PAIR_MIN_DISTANCE and not actin_fibers[j].merged:
                if len(set(actin_fibers[i].xs).intersection(set(actin_fibers[j].xs))) < 4:
                    actin_fibers[i].xs.extend(actin_fibers[j].xs)
                    actin_fibers[i].ys.extend(actin_fibers[j].ys)
                    actin_fibers[i].zs.extend(actin_fibers[j].zs)
                    actin_fibers[i].last_layer.extend(actin_fibers[j].last_layer)
                    actin_fibers[i].cnts.extend(actin_fibers[j].cnts)
                    actin_fibers[i].n += actin_fibers[j].n

                    actin_fibers[j].merged = True

    ax = plt.axes(projection='3d')
    for fiber in actin_fibers:
        if fiber.merged or fiber.n < MIN_FIBER_LENGTH_FINAL: #len(np.unique(fiber.zs)) < 1 - we had this before not sure for what
            continue

        xdata = fiber.xs
        ydata = fiber.ys
        zdata = fiber.zs
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]]*len(xdata), cmap='Greens')
    plt.show()

    return actin_fibers


if __name__ == "__main__":
    img_3d = save_masks_and_get_3d_img()
    actin_fibers = get_3d_plot(img_3d)
    get_actin_stat(actin_fibers)


