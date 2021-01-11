from czifile import CziFile
import cv2
import numpy as np


# To select sub-matrix
# img = img[300:450, 600:750]

def f(x):
    if x > 14000:
        return 255
    else:
        return x / 14000 * 255


if __name__ == '__main__':
    n_slice = 38
    f = np.vectorize(f)

    for i in range(n_slice):
        print(i)
        img = cv2.imread(r'D:\BioLab\img\temp_image\channel_1_number_' + str(i) + '.png', 7)[2650:2810, 3200:3345]

        img_name_norm = r'D:\BioLab\img\temp_image\one_nuclei\nuclei_norm_channel_' + str(1) + '_number_' + str(i) + '.png'
        norm_image = f(img).astype(np.uint8)
        cv2.imwrite(img_name_norm, norm_image)
        print("Finished {}".format(i))



