import os
import cv2
import numpy as np
from numpy import random
from random import randrange
from PIL import Image, ImageOps
import glob


if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\binary_img\Second_labling\0"

    for i, img_path in enumerate(glob.glob(os.path.join(folder_path, "*.png"))):

        img_file = os.path.basename(img_path)
        nucleus_img = Image.open(img_path)
        width, height = nucleus_img.size
        width_shift = 1 / 2 * (width - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))
        height_shift = 1 / 2 * (height - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))
        gamma = [0.90, 0.95, 1, 1, 1, 1.05, 1.1]

        for j in np.arange(8):
            gamma_corrected = Image.fromarray(np.array(255 * (np.array(nucleus_img)/ 255) ** random.choice(gamma), dtype='uint8'))
            # angle_to_rotate = randrange(360)
            angle_to_rotate = random.choice([0, 90, 180, 270])
            aug_img = gamma_corrected.rotate(angle_to_rotate)
            if (random.choice([1, 0])):
                aug_img = ImageOps.flip(aug_img)
            if (random.choice([1, 0])):
                aug_img = ImageOps.mirror(aug_img)
            # cropped_img = aug_img.crop((height_shift, width_shift, height - height_shift, width - width_shift))
            img_file_name = '_aug_' + str(j) + '_' + img_file
            img_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\New_0", img_file_name)
            aug_img.save(img_path_to_save)
            print("Processing " + img_path)

        # # Trying 4 gamma values.
        # for gamma in [0.65, 0.85, 1, 1.25]:
        #     # Apply gamma correction.
        #     gamma_corrected = Image.fromarray(np.array(255 * (np.array(nucleus_img)/ 255) ** gamma, dtype='uint8'))
        #     cropped_img = gamma_corrected.crop((height_shift, width_shift, height - height_shift, width - width_shift))
        #     img_file_name = '_gamma_transformed_' + str(gamma) + '_' + img_file
        #     img_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\Augumentation_gamma", img_file_name)
        #     cropped_img.save(img_path_to_save)


