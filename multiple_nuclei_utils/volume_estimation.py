import os
import glob
import cv2.cv2 as cv2
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm

from multiple_nuclei_utils.cut_nuclei import get_cnt_center


class NucleusData:
    def __init__(self, cnt, scale):
        self.center = get_cnt_center(cnt)
        self.scale = scale
        self.volume_scale = self.scale['x'] * self.scale['y'] * self.scale['z']
        self.volume = cv2.contourArea(cnt) * self.volume_scale
        self.n = 1

    def update_area(self, cnt):
        self.volume += cv2.contourArea(cnt) * self.volume_scale
        self.n += 1

    def __str__(self):
        ret_str = "Center: ({}, {}), Volume: {}, N: {}".format(self.center[0], self.center[1], self.volume, self.n)
        return ret_str


def load_images(img_folder):
    imgs = []
    for img_path in glob.glob(os.path.join(img_folder, "*.png")):
        imgs.append(cv2.imread(img_path)[:, :, 0])

    imgs = np.asarray(imgs, dtype=np.uint8)
    return imgs


def process_layer(layer, nuclei_data, img_3d, scale):
    cnts = cv2.findContours(img_3d[layer, :, :], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    if layer == 0:
        nuclei_data.extend([NucleusData(cnt, scale) for cnt in cnts])
    else:
        for cnt in cnts:
            cnt_center = get_cnt_center(cnt)

            was_added = False
            for nucleus_data in nuclei_data:
                if nucleus_data.center[0] - 20 < cnt_center[0] < nucleus_data.center[0] + 20 and \
                        nucleus_data.center[1] - 20 < cnt_center[1] < nucleus_data.center[1] + 20:
                    nucleus_data.update_area(cnt)
                    was_added = True
                    break

            if not was_added:
                nuclei_data.append(NucleusData(cnt, scale))


def run_volume_estimation(folder_path, output_folder_path, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    if folder_path is None:
        if not os.path.exists('temp/reconstructed_layers'):
            raise RuntimeError("There is no folder {}\nCan't process images".format("temp/reconstructed_layers"))

        if not os.path.exists('temp/volume_data'):
            os.makedirs('temp/volume_data')

        folder_path = 'temp/reconstructed_layers'
        output_folder_path = 'temp/volume_data'

    img_3d = load_images(folder_path)
    layers, h, w = img_3d.shape

    tifffile.imwrite(os.path.join(output_folder_path, 'reconstructed_3d_image.tif'), img_3d)

    scale = {'x': scale_x, 'y': scale_y, 'z': scale_z}
    nuclei_data = []
    for layer in tqdm(range(layers)):
        process_layer(layer, nuclei_data, img_3d, scale)

    # for i, nucleus_data in enumerate(nuclei_data):
    #     print(i)
    #     print(nucleus_data)

    volumes = [nucleus_data.volume for nucleus_data in nuclei_data if (nucleus_data.n > 2 and nucleus_data.volume > 30)]

    print("Total volume of nuclei - {},\nNumber of reconstructed nuclei - {}".format(np.sum(volumes), len(volumes)))
    np.savetxt(os.path.join(output_folder_path, "individual_volumes.cvs"), volumes, delimiter=",")
    plt.hist(volumes)
    plt.show()


if __name__ == '__main__':
    folder_path = r"D:\BioLab\img\big_masks\control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003\Big_mask_5"
    output_folder_path = None
    scale_x, scale_y, scale_z = 0.25, 0.25, 0.5

    run_volume_estimation(folder_path, output_folder_path, scale_x, scale_y, scale_z)
