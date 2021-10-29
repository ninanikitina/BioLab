import cv2.cv2 as cv2
import numpy as np
import math
from single_nucleus_utils.utils import get_cnt_center


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def cnt_length_tests():
    width = 400
    high = 400
    img = np.zeros((width, high), dtype=np.uint8)
    cv2.circle(img, (width // 2, high // 2), min(width, high) // 4, 255, 1)
    # cv2.line(img, (100, 100), (300, 100), 255)
    # cv2.line(img, (300, 100), (300, 300), 255)

    middle_coordinate = (200, 100)
    target_coordinate = (100, 200)
    # cv2.circle(img, target_coordinate, 1, 255, 2)
    # cv2.circle(img, middle_coordinate, 1, 255, 2)

    #  crop image on specific
    # crop_img = img[0:max(middle_coordinate[0], target_coordinate[0]), 0:max(middle_coordinate[1], target_coordinate[1])]
    crop_img = img.copy()
    crop_img[target_coordinate[0]:width - 1, target_coordinate[1]:high - 1] = 0
    crop_img[target_coordinate[1]:high - 1, 0:width - 1] = 0

    cv2.imshow('result', img), cv2.waitKey(0)
    cv2.imshow('result', crop_img), cv2.waitKey(0)

    # find arc lenght
    contours, hierarchy = cv2.findContours(crop_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    print(f"cv2.arcLength(contours[0], closed=False): {cv2.arcLength(contours[0], closed=False)}")
    print(f"cv2.arcLength(contours[0], closed=True): {cv2.arcLength(contours[0], closed=True)}")
    expected = 2 * math.pi * 100 // 4
    print(f"Expected is {expected}")


def cnt_rotation_test():
    img = cv2.imread(r"D:\BioLab\src\single_nucleus_utils\temp\actin_and_nucleus_layers\2channels_3channels_w-Matthew_nucleus_layer_14.png", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)

    _, img_thresh = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(img_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    cnt = max(cnts, key=lambda x: cv2.contourArea(x))

    new_img = rotate_bound(img, cv2.minAreaRect(cnt)[2])

    cv2.imshow("dfdgdfg", new_img)
    cv2.waitKey()


if __name__ == "__main__":
    # cnt_length_tests()
    cnt_rotation_test()
