import os
import glob

from PIL import Image
from tqdm import tqdm
import cv2.cv2 as cv2
from single_nucleus_utils import utils

if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\Big_nucleus_training_img_and_mask\nucleus_trainging\nucleus_512_512_optimized\mask"
    output_folder_path = r'D:\BioLab\img\Big_nucleus_training_img_and_mask\nucleus_trainging\nucleus_512_512_optimized\mask_resized'
    for img_path in tqdm(glob.glob(os.path.join(folder_path, "*.bmp"))):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("cropped", img)

        resized_img = utils.resize_mask(img, (0.04, 0.04, 0.17), "width")
        img_name = "resized_" + os.path.basename(img_path)
        img_path_to_save = os.path.join(output_folder_path, img_name)
        cv2.imwrite(img_path_to_save, resized_img)
