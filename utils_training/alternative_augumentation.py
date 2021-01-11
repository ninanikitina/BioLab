from PIL import Image
import cv2
import numpy as np
from multiple_nuclei_utils.cut_nuclei import get_cnt_center


def get_central_cnt_mask(mask):
    h, w = mask.shape
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    center_cnt = None
    for cnt in cnts:
        center = get_cnt_center(cnt)
        if w // 2 - 50 < center[0] < w // 2 + 50 and h // 2 - 50 < center[1] < h // 2 + 50:
            center_cnt = cnt

    center_cnt_mask = np.zeros_like(mask)
    cv2.drawContours(center_cnt_mask, [center_cnt], -1, 255, -1)
    center_cnt_mask = cv2.dilate(center_cnt_mask, np.ones((5,5),np.uint8))
    return center_cnt_mask


img_pil = Image.open(r"C:\Users\nnina\Desktop\Test_image\norm_channel_1_number_8_934_2818.png")
mask_pil = Image.open(r"C:\Users\nnina\Desktop\Test_mask\norm_channel_1_number_8_934_2818.bmp")

img = np.array(img_pil)
mask = np.array(mask_pil, dtype=np.uint8)
mask = get_central_cnt_mask(mask)

cut_img = np.zeros_like(img)
cut_img[np.where(mask != 0)] = img[np.where(mask != 0)]
img[np.where(mask != 0)] = 0

cut_img_rotated = np.array(Image.fromarray(cut_img).rotate(135))
img[np.where(cut_img_rotated != 0)] = cut_img_rotated[np.where(cut_img_rotated != 0)]



Image.fromarray(img).show()
