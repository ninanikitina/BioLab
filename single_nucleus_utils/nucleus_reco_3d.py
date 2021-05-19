import os
import glob
import matplotlib.pyplot as plt

import cv2.cv2 as cv2
import numpy as np


SCALE_X = 0.04  # units - micrometer
SCALE_Y = 0.04  # units - micrometer
SCALE_Z = 0.04  # units - micrometer (image was resized to account for different scale along z axis)

CENTER_X = 510 // 2  # 510 x_slices in 3d image, take the middle for the center's x coordinate
CENTER_Z = 740


def get_nucleus_origin(nucleus_3d_img):
    center_x = nucleus_3d_img.shape[0] // 2

    slice_cnts = cv2.findContours(nucleus_3d_img[center_x, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]

    left = tuple(slice_cnt[slice_cnt[:, :, 0].argmin()][0])
    right = tuple(slice_cnt[slice_cnt[:, :, 0].argmax()][0])
    top = tuple(slice_cnt[slice_cnt[:, :, 1].argmin()][0])
    bot = tuple(slice_cnt[slice_cnt[:, :, 1].argmax()][0])

    center_y = (bot[1] - top[1]) // 2
    center_z = right[0]

    return center_x, center_y, center_z


def get_nucleus_xsection_cnts(input_folder):
    object_slices = []
    for img_path in glob.glob(os.path.join(input_folder, "*.png")):
        slice_number = int(img_path.rsplit('.', 1)[0].rsplit('_')[-1])

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (1360, 340), interpolation=cv2.INTER_NEAREST)

        object_slices.append([img, slice_number])

    object_slices = sorted(object_slices, key=lambda x: x[1])

    nucleus_3d_img = np.asarray([img for img, layer in object_slices], dtype=np.uint8)

    return nucleus_3d_img


def nucleus_reco_3d(nucleus_3d_img):
    points = []
    center_x, center_y, center_z = get_nucleus_origin(nucleus_3d_img)

    xdata, ydata, zdata = [], [], []
    volume = 0
    for slice in range(nucleus_3d_img.shape[0]):
        xsection_img = nucleus_3d_img[slice, :, :]

        slice_cnts = cv2.findContours(xsection_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]
        volume += cv2.contourArea(slice_cnt) * SCALE_X * SCALE_Y * SCALE_Z

        if slice % 15 == 0:
            ys = [pt[0, 0] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720]
            zs = [pt[0, 1] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720]

            xdata.extend([slice] * len(ys))
            ydata.extend(ys)
            zdata.extend(zs)

        cnt_ys = slice_cnt[:, 0, 1]
        cnt_zs = slice_cnt[:, 0, 0]

        points.extend([[x - center_x, y - center_y, center_z - z] for x, y, z in zip([slice]*len(cnt_ys), cnt_ys, cnt_zs)])

    np.savetxt("nucleus_points_coordinates.csv", np.array(points, dtype=np.int), delimiter=",", fmt="%10.0f")

    print("Nucleus volume: {}".format(volume))

    ax = plt.axes(projection='3d')
    ax.scatter3D(ydata, zdata, xdata, cmap='Greens', alpha=0.5)
    plt.show()


if __name__ == "__main__":
    input_folder = r"C:\Users\nnina\Desktop\prediction_nucleus_mask_e10_w5"

    nucleus_3d_img = get_nucleus_xsection_cnts(input_folder)
    nucleus_reco_3d(nucleus_3d_img)