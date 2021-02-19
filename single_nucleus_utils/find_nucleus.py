import os
import glob
import csv
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt

from multiple_nuclei_utils.cut_nuclei import get_cnts, draw_cnts, get_cnt_center


MIN_FIBER_LENGTH = 30


class ActinFiber(object):
    scale_x = 0.04  # units - micrometer
    scale_y = 0.04  # units - micrometer
    scale_z = 0.04  # units - micrometer (image was resized to account for different scale along z axis)

    def __init__(self, x, y, z, layer, cnt):
        self.xs = [x]
        self.ys = [y]
        self.zs = [z]
        self.cnts = [cnt]
        self.last_layer = [layer]
        self.n = 1

    def update(self, x, y, z, layer, cnt):
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.cnts.append(cnt)
        self.last_layer.append(layer)
        self.n += 1

    def get_stat(self):
        """
        actin_length - full length of fiber including gaps
        actin_xsection - sum of all xsections for each layer times scale
        actin_volume - actin length times average xsection
        """
        actin_length = (self.xs[-1] - self.xs[0]) * self.scale_x
        actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in self.cnts]) * self.scale_y * self.scale_z
        actin_volume = actin_length * actin_xsection

        max_gap_in_layers = 0
        for i in range(self.n - 1):
            if self.xs[i + 1] - self.xs[i] > max_gap_in_layers:
                max_gap_in_layers = self.xs[i + 1] - self.xs[i] - 1

        return [actin_length, actin_xsection, actin_volume, self.n, max_gap_in_layers]


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


def get_3d_image(input_folder, output_folder, type, biggest_layer, biggest_nucleus_mask):
    object_layers = []

    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + type + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        if layer >= biggest_layer:
            object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            object_layer = cv2.bitwise_and(object_img, biggest_nucleus_mask)
            object_layers.append([object_layer, layer])

            actin_cap_path = os.path.join(output_folder, os.path.basename(img_path))
            cv2.imwrite(actin_cap_path, object_layer)

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    actin_3d_image = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    actin_3d_image = np.moveaxis(actin_3d_image, 0, -1)
    return actin_3d_image


def get_all_actin_coords(xsection, x):
    _, mask = cv2.threshold(xsection, 50, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    centers = []
    for cnt in cnts:
        z, y = get_cnt_center(cnt)
        centers.append([x, y + 680, z, cnt])

    cv2.imwrite(os.path.join(output_folder, "xsection_" + str(x) + ".png"), cv2.resize(xsection, (4*mask.shape[1], 4*mask.shape[0])))

    return centers


def get_actin_xsections(actin_3d_img, output_folder):
    x_start, x_end, step = 1850, 2360, 1

    actin_fibers = []
    actin_coords = []
    for x_slice in range(x_start, x_end, step):
        xsection = actin_3d_img[680: 1020, x_slice, :]

        # xsection = cv2.resize(xsection, (340*1, 25*4), interpolation=cv2.INTER_CUBIC)
        current_layer_coords = get_all_actin_coords(xsection, x_slice)

        if x_slice == x_start or not actin_fibers:
            actin_fibers.extend([ActinFiber(x, y, z, x_slice, cnt) for x, y, z, cnt in current_layer_coords])
        else:
            for x, y, z, cnt in current_layer_coords:
                was_added = False
                for actin_fiber in actin_fibers:
                    if actin_fiber.ys[-1] - 8 < y < actin_fiber.ys[-1] + 8 and actin_fiber.zs[-1] - 4 < z < actin_fiber.zs[-1] + 4 and x - actin_fiber.last_layer[-1] < 20:
                        actin_fiber.update(x, y, z, x_slice, cnt)
                        was_added = True
                        break

                if not was_added:
                    actin_fibers.append(ActinFiber(x, y, z, x_slice, cnt))

        actin_coords.extend(current_layer_coords)
        # cv2.imwrite(os.path.join(output_folder, "xsection_" + str(i) + ".png"), xsection)

    ax = plt.axes(projection='3d')
    for fiber in actin_fibers:
        if len(np.unique(fiber.zs)) < 1:
            continue

        xdata = [x for x in fiber.xs if fiber.n > MIN_FIBER_LENGTH]
        ydata = [y for y in fiber.ys if fiber.n > MIN_FIBER_LENGTH]
        zdata = [z for z in fiber.zs if fiber.n > MIN_FIBER_LENGTH]
        color_x = 1.0 * np.random.randint(255) / 255
        color_y = 1.0 * np.random.randint(255) / 255
        color_z = 1.0 * np.random.randint(255) / 255
        if xdata:
            ax.scatter3D(xdata, ydata, zdata, c=[[color_x, color_y, color_z]]*len(xdata), cmap='Greens')
    plt.show()

    return actin_fibers


def get_nucleus_xsections(nucleus_3d_image, actin_fibers):
    x_start, x_end, step = 1850, 2360, 1
    xsection = nucleus_3d_image[680: 1020, x_start + (x_end - x_start) // 2, :]

    draw_img = np.zeros((*xsection.shape, 3), dtype=np.uint8)

    # draw good actin fibers on the middle xsection layer
    actins_to_draw = [actin for actin in actin_fibers if x_start + (x_end - x_start) // 2 in actin.xs and actin.n > 0]
    cnts_to_draw = []
    for actin in actins_to_draw:
        cnts_to_draw.append(actin.cnts[actin.xs.index(x_start + (x_end - x_start) // 2)])
    cv2.drawContours(draw_img, cnts_to_draw, -1, (0, 255, 0), -1)

    # find approximate nucleus contour on the middle xsection layer
    _, mask = cv2.threshold(cv2.blur(xsection, (5, 5)), 45, 255, cv2.THRESH_BINARY)
    nucleus_cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    nucleus_cnt = sorted(nucleus_cnts, key=lambda cnt: len(cnt), reverse=True)[0]
    cv2.drawContours(draw_img, nucleus_cnts, -1, (255, 0, 0), 1)

    size = (draw_img.shape[1] * 8, draw_img.shape[0] * 2)
    draw_img = cv2.resize(draw_img, size, interpolation=cv2.INTER_CUBIC)

    _, mask = cv2.threshold(xsection, 100, 255, cv2.THRESH_BINARY)
    qq = 1


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


if __name__ == "__main__":
    input_folder = 'temp/czi_layers'
    output_folder = 'temp/actin_and_nucleus_layers'

    biggest_layer, biggest_nucleus_mask = find_biggest_nucleus_layer(input_folder)
    actin_3d_img = get_3d_image(input_folder, output_folder, 'actin', 0, biggest_nucleus_mask)
    nucleus_3d_img = get_3d_image(input_folder, output_folder, 'nucleus', 0, biggest_nucleus_mask)
    actin_fibers = get_actin_xsections(actin_3d_img, output_folder)

    get_actin_stat(actin_fibers)

    #get_nucleus_xsections(nucleus_3d_img, actin_fibers)
