import glob
import os
import cv2.cv2 as cv2
import numpy as np

from single_nucleus_utils.structures import CntExtremes

def get_cnt_extremes(cnt):
    """
    Finds contour extremes
    ---
        Parameters:
        cnt (vector<std::vector<cv::Point>>): contour which is a vector of points.
    ---
        Returns:
        cnt_extremes (CntExtremes object): where left, right, top, bottom attributes are coordinates
                                        of the corresponding extreme points of the specified contour
    """
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

    return CntExtremes(left, right, top, bottom)


def get_cnts(img, threshold):
    """
    Finds and process (remove noise, smooth edges) all contours on the specified image
    ---
        Parameters:
        threshold (int): threshold of pixel intensity starting from which pixels will be labeled as 1 (white)
        img (image): image
    ---
        Returns:
        cnts (vector<std::vector<cv::Point>>): List of detected contours.
                            Each contour is stored as a vector of points.
    """
    _, img_thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    # Apply pair of morphological "opening" and "closing" to remove noise and to smoothing edges
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((5, 5)))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((5, 5)))

    cnts = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    if len(cnts) == 0:
        return None

    hulls = [cv2.convexHull(cnt, False) for cnt in cnts]
    mask = np.zeros_like(img_thresh)
    cv2.drawContours(mask, hulls, -1, 255, -1)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    return cnts


def get_cnt_center(cnt):
    if len(cnt) <= 2:
        center = (cnt[0, 0, 0], cnt[0, 0, 1])
    else:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            center_x = int(np.mean([cnt[i, 0, 0] for i in range(len(cnt))]))
            center_y = int(np.mean([cnt[i, 0, 1] for i in range(len(cnt))]))
        else:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])

        center = (center_x, center_y)

    return center


def draw_cnts(shape, cnts):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, cnts, -1, color=255, thickness=-1)

    return mask

def prepare_folder(output_folder):
    """
    Creates folder if it has not been created before
    and cleans the folder
    ---
        Parameters:
        output_folder (string): folder's path
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for f in glob.glob(output_folder + "/*"):
        os.remove(f)


def make_padding(img, final_img_size):
    h, w = img.shape[:2]
    h_out, w_out = final_img_size

    top = (h_out - h) // 2
    bottom = h_out - h - top
    left = (w_out - w) // 2
    right = w_out - w - left

    padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded_img


def resize_mask(img, scale, axis):
    if axis == "width":
        dsize = (int(img.shape[1] * (scale[2] / scale[0])), img.shape[0])  # img.shape = (width, height)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
        crop_img = img[:, img.shape[1]//2 - 512 // 2:  img.shape[1]//2 + 512 // 2]
    if axis == "height":
        dsize = (int(img.shape[1] , img.shape[0] * (scale[2] / scale[0])))  # img.shape = (width, height)
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_NEAREST)
        crop_img = img[img.shape[0] // 2 - 512 // 2:  img.shape[0] // 2 + 512 // 2 , :]
    return crop_img


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
