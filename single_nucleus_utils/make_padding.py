import os
import glob
from tqdm import tqdm
import cv2.cv2 as cv2
import numpy as np

input_folder = r"D:\BioLab\Tests_for_hand_labling\Image_of_all_layers"
output_folder = r"D:\BioLab\Tests_for_hand_labling\Image_of_all_layer_with_padding"

if __name__ == "__main__":
    for img_path in glob.glob(input_folder + r"\*"):
        img_name= os.path.basename(img_path)

        img = cv2.imread(img_path, 0)
        #img = np.flip(img, axis=0)
        desired_size = 1360

        old_size = img.shape[:2]  # old_size is in (height, width) format

        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        im = cv2.resize(img, (new_size[1], new_size[0]))

        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        # cv2.imshow("image", new_im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        new_img_path = os.path.join(output_folder, img_name)
        cv2.imwrite(new_img_path, new_im)



