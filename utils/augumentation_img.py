import os
import cv2
import numpy as np
from random import randrange
from PIL import Image
import glob


if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\unet_imgs\train\IMG_new"
    mask_folder_path =r"D:\BioLab\img\unet_imgs\train\mask_NEW"

    for i, img_path in enumerate(glob.glob(os.path.join(folder_path, "*.png"))):
        mask_path = glob.glob(os.path.join(mask_folder_path, "*.bmp"))[i]

        img_file = os.path.basename(img_path)
        mask_file = os.path.basename(mask_path)
        nucleus_img = Image.open(img_path)
        nucleus_mask = Image.open(mask_path)
        width, height = nucleus_img.size
        width_shift = 1 / 2 * (width - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))
        height_shift = 1 / 2 * (height - ((height / 2) ** 2 + (width / 2) ** 2) ** (1 / 2))

        for j in np.arange(10):
            angle_to_rotate = randrange(360)
            rotated_img = nucleus_img.rotate(angle_to_rotate)
            cropped_img = rotated_img.crop((height_shift, width_shift, height - height_shift, width - width_shift))
            rotated_mask = nucleus_mask.rotate(angle_to_rotate)
            cropped_mask = rotated_mask.crop((height_shift, width_shift, height - height_shift, width - width_shift))
            img_file_name = '_rotated_' + str(j) + '_' + img_file
            mask_file_name = '_rotated_' + str(j) + '_' + mask_file
            img_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\Processed_image", img_file_name)
            mask_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\Processed_mask", mask_file_name)
            cropped_img.save(img_path_to_save)
            cropped_mask.save(mask_path_to_save)
            print("Processing " + img_path)

        for k in np.arange(10):
            random_left_shift = randrange((int)(width_shift * 2))
            random_up_shift = randrange((int)(height_shift * 2))
            right_shift = width - (2 * width_shift - random_left_shift)
            bottom_shift = height - (2 * height_shift - random_up_shift)
            cropped_shifted_img = nucleus_img.crop((random_up_shift, random_left_shift, bottom_shift, right_shift))
            cropped_shifted_mask = nucleus_mask.crop((random_up_shift, random_left_shift, bottom_shift, right_shift))
            img_file_name = '_translated_' + str(k) + '_' + img_file
            mask_file_name = '_translated_' + str(k) + '_' + mask_file
            img_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\Processed_image", img_file_name)
            mask_path_to_save = os.path.join(r"C:\Users\nnina\Desktop\Processed_mask", mask_file_name)
            cropped_shifted_img.save(img_path_to_save)
            cropped_shifted_mask.save(mask_path_to_save)









