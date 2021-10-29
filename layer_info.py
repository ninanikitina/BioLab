import os
import glob
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

from single_nucleus_utils import utils
from single_nucleus_utils.utils import draw_cnts
from single_nucleus_utils.structures import ActinFiber, ActinContour, ConfocalImgReader, CntExtremes
from single_nucleus_utils.node_creation import run_node_creation

from unet.predict import run_predict_unet


CONFOCAL_IMG = r"D:\BioLab\img\3Dimg_for_tests\3D\63x\control siRNA 63x_3x1_5_um_stack_11_16_Airyscan_Processing_Stitch.czi"

NUCLEUS_CHANNEL = 1
ACTIN_CHANNEL = 0
NOISE_TH = 0.0001
CNT_AREA_TH = 700
ACTIN_UNET_MODEL = r"D:\BioLab\models\actin\CP_epoch200_actin_weight.corection_200_labling_V2.pth"
NUCLEUS_UNET_MODEL = r"D:\BioLab\models\one_nucleus\CP_epoch200_nucleus_weight.corection_200_labling_V2_512_512_no_agum.pth"
UNET_MODEL_SCALE = 1
UNET_IMG_SIZE = (512, 512)
UNET_MODEL_THRESHOLD = 0.5
SCALE_X = 0.04
SCALE_Y = 0.04
SCALE_Z = 0.17
MIN_FIBER_LENGTH = 20


file_to_read = open(r"D:\BioLab\src\single_nucleus_utils\actin_data_long2.obj",
                        "rb")  # change back to "actin-data_long.obj"
actin_fibers = pickle.load(file_to_read)
actin_fibers = [actin for actin in actin_fibers if actin.n > MIN_FIBER_LENGTH]

#getting center coordinates from
# center_x, center_y, center_z = get_nucleus_origin(nucleus_3d_img)
center_x, center_y, center_z = 249, 256, 271

areas = []
new_cnts = []
y_cnt, z_cnt = [], []
for actin in actin_fibers:
    if center_x in actin.xs:
        idx = actin.xs.index(center_x)
        y_cnt.append(actin.ys[idx])
        z_cnt.append(actin.zs[idx])
        mask = draw_cnts((512, 512), [actin.cnts[idx]])
        areas.append(np.count_nonzero(mask) * SCALE_Y * SCALE_Z)
        new_cnts.append(actin.cnts[idx])

data = []
data.extend([[(y - center_y) * SCALE_X, (center_z - z) * SCALE_Z, s] for y, z, s in zip(y_cnt, z_cnt, areas) if (center_z - z) > 2])

np.savetxt("middle_layera_actin_points_coordinates.csv", np.array(data, dtype=np.double), delimiter=",", fmt="%10.2f")
for i, _ in enumerate(z_cnt):
    print(f"y = {y_cnt[i]} z = {z_cnt[i]}, area = {areas[i]}")

mask = draw_cnts((512, 512), new_cnts)

cv2.imshow("dfsdf", mask)
cv2.waitKey()
print("test")





