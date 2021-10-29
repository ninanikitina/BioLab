import os
import sys

import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm

from czifile import CziFile
from readlif.reader import LifFile


class ActinContour(object):
    def __init__(self, x, y, z, cnt):
        self.x = x
        self.y = y
        self.z = z
        self.cnt = cnt
        self.xsection = 0
        self.parent = None


class ActinFiber(object):
    # scale_x = 0.04  # units - micrometer
    # scale_y = 0.04  # units - micrometer (image was resized to account for different scale along z axis)
    # scale_z = 0.17  # units - micrometer (image was resized to account for different scale along z axis)

    def __init__(self, x, y, z, layer, cnt):
        self.xs = [x]
        self.ys = [y]
        self.zs = [z]
        self.cnts = [cnt]
        self.last_layer = [layer]
        self.n = 1
        self.line = None
        self.merged = False

    def update(self, x, y, z, layer, cnt):
        self.xs.append(x)
        self.ys.append(y)
        self.zs.append(z)
        self.cnts.append(cnt)
        self.last_layer.append(layer)
        self.n += 1

    def fit_line(self):
        points = np.asarray([[x, y] for x, y in zip(self.xs, self.ys)])
        self.line = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

    def get_stat(self, scale_x, scale_y, scale_z, is_rescaled):
        """
        actin_length - full length of fiber including gaps
        actin_xsection - sum of all xsections for each layer times scale
        actin_volume - actin length times average xsection
        """
        actin_length = (self.xs[-1] - self.xs[0]) * scale_x
        if is_rescaled:
            actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in self.cnts]) * scale_y ** 2
        else:
            actin_xsection = np.mean([cv2.contourArea(cnt) for cnt in self.cnts]) * scale_y * scale_z
        actin_volume = actin_length * actin_xsection

        max_gap_in_layers = 0
        for i in range(self.n - 1):
            if self.xs[i + 1] - self.xs[i] > max_gap_in_layers:
                max_gap_in_layers = self.xs[i + 1] - self.xs[i] - 1

        return [actin_length, actin_xsection, actin_volume, self.n, max_gap_in_layers]


class ConfocalImgReader(object):
    """
    Creates an object that reads confocal microscopy images of two channels (actin and nucleus)
    """

    def __init__(self, img_path, noise_th, nucleus_channel, actin_channel):
        """
            Parameters:
            img_path (string): path to the file to read
            noise_th (float): noise normalization threshold
            nucleus_channel(int): channel of nucleus images at the provided microscopic image
            actin_channel(int): channel of actin images at the provided microscopic image
        """
        self.img_path = img_path
        _, self.ext = os.path.splitext(img_path)
        self.img_name = os.path.splitext(os.path.basename(img_path))[0]
        self.noise_tr = noise_th
        self.nuclei_channel = nucleus_channel
        self.actin_channel = actin_channel
        self.type, self.image_arrays = None, None

    def _read_as_czi(self):
        with CziFile(self.img_path) as czi:
            self.image_arrays = czi.asarray()
        self.type = self.image_arrays[0, 0, 0, 0, :, :, 0].dtype

    def _read_as_lif(self):
        lif = LifFile(self.img_path)
        nucleus_array = np.asarray([np.array(pil_image) * 16 for pil_image in lif.get_image(0).get_iter_z(t=0, c=0)]) #lif.get_image(0) - image num
        actin_array = np.asarray([np.array(pil_image) * 16 for pil_image in lif.get_image(0).get_iter_z(t=0, c=1)])
        self.image_arrays = np.stack([nucleus_array, actin_array], axis=3)
        self.image_arrays = np.moveaxis(self.image_arrays, 0, 2)
        self.type = self.image_arrays.dtype

    def _normalization(self, img, norm_th, pixel_value):
        """
        normalized specified image based on normalization threshold
        ---
        Parameters:
            img (array): array of pixels form 0 to 255 for 8-bit image or
                        form 0 to 65535 for 16-image to be normalized
            norm_th(float): normalization threshold
            pixel_value:
        """
        img[np.where(img > (pixel_value - 1) - norm_th)] = (pixel_value - 1) - norm_th
        img = cv2.normalize(img, None, alpha=0, beta=(pixel_value - 1), norm_type=cv2.NORM_MINMAX)
        img = (img / 256).astype(np.uint8)

        return img

    def _get_img_path(self, img_name, layer, output_folder):
        img_path_norm = os.path.join(output_folder, img_name + '_layer_' + str(layer) + '.png')

        return img_path_norm

    def _get_norm_th(self, pixel_value):
        middle_layer = self.image_arrays.shape[2] // 2  #TODO this channel is different for different image formats
        img = self.image_arrays[:, :, middle_layer, self.actin_channel]
        hist = np.squeeze(cv2.calcHist([img], [0], None, [pixel_value], [0, pixel_value]))
        norm_th, sum = 1, 0
        while (sum / img.size < self.noise_tr):
            sum += hist[-norm_th]
            norm_th += 1

        return norm_th

    def _save_normalized_img(self, bio_structure, norm_th, pixel_value, output_folder):
        for i in tqdm(range(self.image_arrays.shape[2])):
            img_path_norm = self._get_img_path(self.img_name + "_" + bio_structure, i, output_folder)
            channel = self.actin_channel if bio_structure == "actin" else self.nuclei_channel
            norm_image = self._normalization(self.image_arrays[:, :, i, channel], norm_th, pixel_value)
            cv2.imwrite(img_path_norm, norm_image)
        # print("\n{} image has {} layers of shape (h={}, w={})\n".format(bio_structure,
        #                                                                 self.image_arrays.shape[4],
        #                                                                 self.image_arrays.shape[5],
        #                                                                 self.image_arrays.shape[6]))

    def _save_img(self, bio_structure, output_folder):
        for i in tqdm(range(self.image_arrays.shape[2])):
            img_path = self._get_img_path(self.img_name + "_" + bio_structure, i, output_folder)
            channel = self.actin_channel if bio_structure == "actin" else self.nuclei_channel
            cv2.imwrite(img_path, np.uint8(self.image_arrays[:, :, i, channel] / 256))  # convert to 8-bit image

    def read(self, output_folder):
        """
        Converts confocal microscopic images into a set of jpg images specified in reader object normalization
        ---
            Parameters:
            output_folder (string): path to the folder to save jpg images
        """
        if self.ext == ".lif":
            self._read_as_lif()
            self._save_img("actin", output_folder)
            self._save_img("nucleus", output_folder)
            # pixel_value = 65536 if self.type == "uint16" else 256
            # norm_th = self._get_norm_th(pixel_value)
            # self._save_normalized_img("actin", norm_th, pixel_value, output_folder)
            # self._save_normalized_img("nucleus", norm_th, pixel_value, output_folder)

        elif self.ext == ".czi":
            self._read_as_czi()
            pixel_value = 65536 if self.type == "uint16" else 256
            norm_th = self._get_norm_th(pixel_value)
            self._save_normalized_img("actin", norm_th, pixel_value, output_folder)
            self._save_normalized_img("nucleus", norm_th, pixel_value, output_folder)
        else:
            print("File is not in czi or lif format!")
            sys.exit()


class CntExtremes(object):
    def __init__(self, left, right, top, bottom):
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom
