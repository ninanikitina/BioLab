import tifffile
import os
import glob
import cv2.cv2 as cv2
import numpy as np

folder_path = r"C:\Users\nnina\Desktop\actin layers"
output_folder_path =r"C:\Users\nnina\Desktop\3d"


def load_images(img_folder):
    imgs = []
    for img_path in glob.glob(os.path.join(img_folder, "*.png")):
        imgs.append(cv2.imread(img_path)[:, :, 0])

    imgs = np.asarray(imgs, dtype=np.uint8)
    return imgs


img_3d = load_images(folder_path)
layers, h, w = img_3d.shape

tifffile.imwrite(os.path.join(output_folder_path, 'reconstructed_3d_image.tif'), img_3d)