from czifile import CziFile
import cv2
import numpy as np


# Typical shape is (1, 1, 2, 1, 38, 6355, 6359, 1)

def f(x):
    if x > 15000:
        return np.uint8(255)
    else:
        return np.uint8(x / 15000 * 255)


if __name__ == '__main__':
    f = np.vectorize(f)

    with CziFile(r"D:\BioLab\img\img_for_tests\3D\20x\control siRNA 20x 5x5 15um stack_Airyscan Processing_Stitch-003.czi") as czi:
        image_arrays = czi.asarray()

    for i in range(image_arrays.shape[4]):
       # for j in range(2):
        img_name_norm = r'D:\BioLab\img\temp_image\norm_channel_' + str(1) + '_number_' + str(i) + '.png'
        norm_image = f(image_arrays[0, 0, 1, 0, i , :, :, 0]).astype(np.uint8)
        cv2.imwrite(img_name_norm, norm_image)

    print(image_arrays.shape)


def func():
    pass