import os
import glob
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
import numpy as np
from PIL import Image

from multiprocessing import Pool


if __name__ == "__main__":
    input_folder = r"D:\BioLab\src\single_nucleus_utils\temp\actin_and_nucleus_layers"
    identifier = "actin"

    object_layers = []
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layers.append([object_img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    max_progection = image_3d.max(axis=0, out=None, keepdims=False,  where = True)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", max_progection)
    cv2.waitKey()
    # Find the edges in the image using canny detector
    edges = cv2.Canny(max_progection, 50, 200)

    # Detect points that form a line
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold = 15, minLineLength=10, maxLineGap=250)

    # Draw lines on the image
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(max_progection, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angles.append(np.arctan((y2-y1)/(x2 - x1)) * 180/np.pi)
    # Show result
    # Create window with freedom of dimensions
    print(np.mean(angles))
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", max_progection)
    cv2.waitKey()

    #image_3d = np.moveaxis(image_3d, 0, -1)
    a = 1


