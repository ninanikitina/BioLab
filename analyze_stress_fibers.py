import os
import glob
import cv2.cv2 as cv2
import numpy as np
from czifile import CziFile
from tqdm import tqdm
import sys

from single_nucleus_utils.utils import prepare_folder
from single_nucleus_utils.czi_read_one_nucleus_255 import run_czi_reader_255

from multiple_nuclei_utils.cut_nuclei import run_cut_nuclei
from multiple_nuclei_utils.reconstruct_layers import merge_individual_masks
from multiple_nuclei_utils.volume_estimation import run_volume_estimation
from unet.predict import run_predict_unet


CONFOCAL_IMG = r"D:\BioLab\img\3Dimg_for_tests\3D\63x\control siRNA 63x_3x1_5_um_stack_11_16_Airyscan_Processing_Stitch.czi"
#CONFOCAL_IMG = r"D:\BioLab\img\Buer_img\Test1-L4-0.12.czi"

NUCLEI_CHANNEL = 1
ACTIN_CHANNEL = 0
NOISE_TH = 0.0001
BITS = 16  # automate checking channgel size
CNT_AREA_TH = 700
UNET_MODEL = r"D:\BioLab\models\CP_epoch15_unet_no_data-aug.pth"
UNET_MODEL_SCALE = 1
UNET_IMG_SIZE = (256, 256)
UNET_MODEL_THRESHOLD = 0.5
SCALE_X = 0.04
SCALE_Y = 0.04
SCALE_Z = 0.17
RESCALE_Z_FACTOR = SCALE_Z / SCALE_Y  # helps to make image in 1:1 scale

# -------------------------------------------------------
class ConfocalImgReader():
    def __init__(self, img_path, output_folder, noise_th, nuclei_channel, actin_channel):
        self.img_path = img_path
        self.output_folder = output_folder
        _, self.ext = os.path.splitext(img_path)
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.noise_tr = noise_th
        self.nuclei_channel = nuclei_channel
        self.actin_channel = actin_channel
        self.type, self.image_arrays = None, None


    def _read_as_czi(self):
        with CziFile(self.img_path) as czi:
            self.image_arrays = czi.asarray()
        self.type = self.image_arrays[0, 0, 0, 0, :, :, 0].dtype


    def _normalization(self, img, norm_th, pixel_value):
        img[np.where(img > (pixel_value - 1) - norm_th)] = (pixel_value - 1) - norm_th
        img = cv2.normalize(img, None, alpha=0, beta=(pixel_value - 1), norm_type=cv2.NORM_MINMAX)
        img = (img / 256).astype(np.uint8)

        return img


    def _get_img_path(self, img_name, layer):
        img_path_norm = os.path.join(self.output_folder, img_name + '_layer_' + str(layer) + '.png')

        return img_path_norm


    def _get_norm_th(self, pixel_value):
        middle_layer = self.image_arrays.shape[4] // 2
        img = self.image_arrays[0, 0, self.nuclei_channel, 0, middle_layer, :, :, 0]
        hist = np.squeeze(cv2.calcHist([img], [0], None, [pixel_value], [0, pixel_value]))
        norm_th, sum = 1, 0
        while (sum / img.size < self.noise_tr):
            sum += hist[-norm_th]
            norm_th += 1

        return norm_th


    def _save_normalized_img(self, nucleus_or_actin, norm_th, pixel_value):
        for i in tqdm(range(self.image_arrays.shape[4])):
            img_path_norm = self._get_img_path(self.img_name + "_" + nucleus_or_actin, i)
            channel = self.actin_channel if nucleus_or_actin == "actin" else self.nuclei_channel
            norm_image = self._normalization(self.image_arrays[0, 0, channel, 0, i, :, :, 0], norm_th, pixel_value)
            cv2.imwrite(img_path_norm, norm_image)
        print("\n{} image has {} layers of shape (h={}, w={})\n".format(nucleus_or_actin,
                                                                   self.image_arrays.shape[4],
                                                                   self.image_arrays.shape[5],
                                                                   self.image_arrays.shape[6]))


    def read(self):
        if self.ext == ".czi":
            self._read_as_czi()
        else:
            print("Can not read non czi files!")
            sys.exit()
        pixel_value = 65536 if self.type == "uint16" else 256
        norm_th = self._get_norm_th(pixel_value)
        self._save_normalized_img("actin", norm_th, pixel_value)
        self._save_normalized_img("nucleus", norm_th, pixel_value)


# -------------------------------------------------------


def draw_cnts(shape, cnts):
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, cnts, -1, color=255, thickness=-1)

    return mask


class CntExtremes():
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


def get_cnt_extremes(cnt):
    left = tuple(cnt[cnt[:, :, 0].argmin()][0])
    right = tuple(cnt[cnt[:, :, 0].argmax()][0])
    top = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottom = tuple(cnt[cnt[:, :, 1].argmax()][0])

    return CntExtremes(left, right, top, bottom)


def get_cnts(img):
    _, img_thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)

    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, np.ones((5, 5)))  #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))
    img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, np.ones((5, 5))) #cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5)))

    cnts, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cnts = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)
    cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > CNT_AREA_TH]
    if len(cnts) == 0:
        return None
    return cnts


def get_nucleus_cnt(img):
    cnts = get_cnts((img))
    if cnts is None:
        return None
    cnt = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]
    return cnt


def find_biggest_nucleus_layer(input_folder):
    nucleus_area = 0
    biggest_nucleus_mask, cnt_extremes = None, None
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_nucleus_*.png"))):
        nucleus_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        current_nucleus_cnt = get_nucleus_cnt(nucleus_img)
        if current_nucleus_cnt is None:
            continue
        current_nucleus_cnt_area = cv2.contourArea(current_nucleus_cnt)
        if current_nucleus_cnt_area > nucleus_area:
            nucleus_area = current_nucleus_cnt_area
            biggest_nucleus_mask = draw_cnts(nucleus_img.shape[:2], [current_nucleus_cnt])
            cnt_extremes = get_cnt_extremes(current_nucleus_cnt)

    return biggest_nucleus_mask, cnt_extremes


def get_3d_image(input_folder, output_folder, type, biggest_nucleus_mask):
    object_layers = []

    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + type + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layer = cv2.bitwise_and(object_img, biggest_nucleus_mask)
        object_layers.append([object_layer, layer])

        cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), object_layer)

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    image_3d = np.moveaxis(image_3d, 0, -1)
    return image_3d

def resize_for_unet(img):
    resized = cv2.resize(img, UNET_IMG_SIZE, interpolation=cv2.INTER_AREA)
    return resized

def make_padding(img):
    color = [0, 0, 0]
    size = img.shape[0] if img.shape[0]%2 == 0 else img.shape[0]+1
    padding = (size - img.shape[1]) // 2
    top, bottom, left, right = 0, 0, padding, padding
    img_w_padding = cv2.copyMakeBorder(img, size - img.shape[0], bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img_w_padding

def get_yz_xsection(img_3d, output_folder, type, cnt_extremes):
    x_start, x_end, step = cnt_extremes.left[0], cnt_extremes.right[0], 1
    for x_slice in range(x_start, x_end, step):
        top = cnt_extremes.top[1] - 20
        bottom = cnt_extremes.bottom[1] + 20
        xsection = img_3d[top: bottom, x_slice, :]
        img_path = os.path.join(output_folder, "xsection_" + type + "_" + str(x_slice) + ".png")
        img = cv2.resize(xsection,(int(RESCALE_Z_FACTOR * xsection.shape[1]), xsection.shape[0]))
        img_w_padding = make_padding(img)
        img_unet_optimized = resize_for_unet(img_w_padding)
        cv2.imwrite(img_path, img_unet_optimized)
    return img_w_padding.shape[1] / UNET_IMG_SIZE[0]


def main():

    print("\nRunning czi reader ...")
    output_folder = "temp/czi_layers"
    prepare_folder(output_folder)
    reader = ConfocalImgReader(CONFOCAL_IMG, output_folder, NOISE_TH, NUCLEI_CHANNEL, ACTIN_CHANNEL)
    reader.read()

    print("\nGenerate xsection images ...")
    input_folder = 'temp/czi_layers'
    output_folder = 'single_nucleus_utils/temp/actin_and_nucleus_layers'
    prepare_folder(output_folder)
    biggest_nucleus_mask, cnt_extremes = find_biggest_nucleus_layer(input_folder)
    actin_3d_img = get_3d_image(input_folder, output_folder, 'actin', biggest_nucleus_mask)
    nucleus_3d_img = get_3d_image(input_folder, output_folder, 'nucleus', biggest_nucleus_mask)

    output_folder = 'single_nucleus_utils/temp/actin_layers'
    prepare_folder(output_folder)
    rescale_unet_factor = get_yz_xsection(actin_3d_img, output_folder, "actin", cnt_extremes)
    print(rescale_unet_factor)

    output_folder = 'single_nucleus_utils/temp/nucleus_layers'
    prepare_folder(output_folder)
    get_yz_xsection(nucleus_3d_img, output_folder, "nuclues", cnt_extremes)

    # run_rotate_layers (find_nuclei + find_actin(part that cat area of interests) + add padding)

    # print("\nRunning unet predictor for actin...")
    # run_predict_unet(None, None, UNET_MODEL, UNET_MODEL_SCALE, UNET_MODEL_THRESHOLD)
    #
    # print("\nRunning unet predictor for nuclei...")
    # run_predict_unet(None, None, UNET_MODEL, UNET_MODEL_SCALE, UNET_MODEL_THRESHOLD)
    #
    # print("\nFinding actin fibers...")
    # # analyze unet ourput mask. Combine together individul dots to create long actin fiber
    #
    # print("\nFinding nuclei shape")
    # # gets dots coordinates for nucleous
    #
    # print("\nRunning node creation and getting statistis data ...")
    # # creates node for nucleous


if __name__ == '__main__':
    main()
