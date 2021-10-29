import glob
import os
import cv2.cv2 as cv2
import numpy as np
import os, glob


ulpath = r"D:/BioLab/img/Big_nucleus_training_img_and_mask/Actin__training_img_and_masks/34_layers_masks_V2/"

img_path = r"D:\BioLab\img\Big_nucleus_training_img_and_mask\Actin__training_img_and_masks\Training_set_with_multiple_cells_512-512\initial_img\mask_512"

for infile in glob.glob( os.path.join(ulpath, "*.bmp")):
    im = cv2.imread(infile, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(im, (512, 512))
    cv2.imwrite(os.path.join(img_path, os.path.basename(infile)), resized_image)
