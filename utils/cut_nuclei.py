import os
import glob
import numpy as np
import cv2


CNT_AREA_TH = 700


def get_cnt_center(cnt):
    M = cv2.moments(cnt)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    center = (center_x, center_y)

    return center


def get_cnts(img):
    _, img_thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((5, 5)))  #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((5, 5))) #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))

    cnts, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > CNT_AREA_TH]

    return cnts


def draw_cnts(shape, cnts):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, cnts, -1, color=255, thickness=-1)

    return mask


def cut_nucleus(img, cnt):
    h, w = img.shape

    center = get_cnt_center(cnt)

    x1, x2 = center[0] - 256, center[0] + 256
    y1, y2 = center[1] - 256, center[1] + 256

    if x1 < 0:
        x1, x2 = 0, 512
    if x2 >= w:
        x1, x2 = w - 1 - 512, w - 1

    if y1 < 0:
        y1, y2 = 0, 512
    if y2 >= h:
        y1, y2 = h - 1 - 512, h - 1

    nucleus_img = img[y1: y2, x1: x2]

    return nucleus_img


def process_img(img_path, output_img_path, output_mask_path):
    file_name = os.path.basename(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    cnts = get_cnts(img)
    mask = draw_cnts(img.shape, cnts)

    for cnt in cnts:
        center = get_cnt_center(cnt)
        nucleus_img = cut_nucleus(img, cnt)
        nucleus_mask = cut_nucleus(mask, cnt)

        name = file_name.rsplit('.', 1)[0] + "_" + str(center[0]) + "_" + str(center[1]) + ".png"

        nucleus_img_path = os.path.join(output_img_path, name)
        nucleus_mask_path = os.path.join(output_mask_path, name)

        cv2.imwrite(nucleus_img_path, nucleus_img)
        cv2.imwrite(nucleus_mask_path, nucleus_mask)


if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\temp_image"
    output_img_path = r"D:\BioLab\img\model_training_img\Big_img_5K_and_mask\control siRNA 20x 5x5 15um stack_Airyscan Processing_Stitch-003\imgs"
    output_mask_path = "D:\BioLab\img\model_training_img\Big_img_5K_and_mask\control siRNA 20x 5x5 15um stack_Airyscan Processing_Stitch-003\masks"

    for img_path in glob.glob(os.path.join(folder_path, "*.png")):
        print("Processing " + img_path)

        process_img(img_path, output_img_path, output_mask_path)
