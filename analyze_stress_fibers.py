import os
import glob
import cv2.cv2 as cv2
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import trimesh
from skimage import measure
import time


from single_nucleus_utils import utils
from single_nucleus_utils.structures import ActinFiber, ActinContour, ConfocalImgReader, CntExtremes
from single_nucleus_utils.node_creation import run_node_creation
from single_nucleus_utils.node_creation import get_actin_stat
from unet.predict import run_predict_unet


#CONFOCAL_IMG = r"D:\BioLab\img\3Dimg_for_tests\3D\63x\control siRNA 63x_3x1_5_um_stack_11_16_Airyscan_Processing_Stitch.czi"
CONFOCAL_IMG = r"D:\BioLab\img\3D_new_set\2021.10.27_MSC_cnt_Actin-488_nuclei-405_63x.lif"
# CONFOCAL_IMG = r"D:\BioLab\img\3D_new_set\2channels_3channels_w-Matthew.lif"

#CONFOCAL_IMG = r"D:\BioLab\img\Buer_img\Test1-L4-0.12.czi"

NUCLEUS_CHANNEL = 0  # 1 for original czi file
ACTIN_CHANNEL = 1  # 0 for original czi file
NOISE_TH = 0.0001

NUCLEUS_CNT_TH = 20 # 30 for original czi file
MIN_FIBER_LENGTH = 20
ACTIN_UNET_MODEL = r"D:\BioLab\models\actin\CP_epoch200_actin_weight.corection_200_labling_V2.pth"
NUCLEUS_UNET_MODEL = r"D:\BioLab\models\one_nucleus\CP_epoch200_nucleus_weight.corection_200_labling_WITH_FLAT_V2_512_512_no_agum_V2.pth"
ACTIN_OBJECT = "single_nucleus_utils/actin_data_long2.obj"
UNET_MODEL_SCALE = 1
UNET_IMG_SIZE = (512, 512)
UNET_MODEL_THRESHOLD = 0.5
SCALE_X = 0.05871
SCALE_Y = 0.05871
SCALE_Z = 0.27
IS_RESCALED = True
NEW_ACTIN_LEN_TH = 5


def rotate_image(input_folder, identifier, rot_angle):
    object_layers = []
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_img = utils.rotate_bound(object_img, rot_angle)
        object_layers.append([object_img, layer])
        # cv2.imshow("output", cv2.resize(object_img, (1000, 1000)))
        # cv2.waitKey()
    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)
    image_3d = np.moveaxis(image_3d, 0, -1)

    return image_3d


def get_nucleus_cnt(img):
    cnts = utils.get_cnts((img), NUCLEUS_CNT_TH)
    if cnts is None:
        return None
    cnt = sorted(cnts, key=lambda cnt: cv2.contourArea(cnt), reverse=True)[0]
    return cnt


def get_cnt_extremes(biggest_nucleus_mask, rot_angle):

    biggest_nucleus_mask = utils.rotate_bound(biggest_nucleus_mask, rot_angle)
    # mask_cnt = cv2.findContours(biggest_nucleus_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0]
    mask_cnt = cv2.findContours(biggest_nucleus_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
    cnt_extremes = utils.get_cnt_extremes(mask_cnt)

    return cnt_extremes

def find_rotation_angle(input_folder):
    """
    Finds rotation angle based on actin fibers orientation
    ---
        Parameters:
        - input_folder (string): path to the folder where all slices of the nucleus
                            in jpg format is located
    ---
        Returns:
        - rot_angle (int): the degree to rotate an image that actin fiber will be
                        parallel to x-axis

    """
    ROTATION_TRH = 250
    identifier = "actin"
    object_layers = []
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layers.append([object_img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    max_progection = image_3d[15:16, :, :].max(axis=0, out=None, keepdims=False,  where = True)   # TODO: now tacking only upper part
    cv2.imshow("output", cv2.resize(max_progection,(1000, 1000)))
    cv2.waitKey()
    # Find the edges in the image using canny detector
    edges = cv2.Canny(max_progection, ROTATION_TRH, 200)

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
    rot_angle = (np.mean(angles))
    cv2.imshow("output", cv2.resize(max_progection,(1000, 1000)))
    cv2.waitKey()

    return rot_angle

def find_biggest_nucleus_layer(input_folder):
    """
    Finds and analyzes image (layer) with the biggest area of the nucleus
    ---
        Parameters:
        - input_folder (string): path to the folder where all slices og the nucleus
                            in jpg format is located
    ---
        Returns:
        - biggest_nucleus_mask (np. array): array of 0 and 1 where 1 is white pixels
                                        which represent the shape of the biggest area
                                        of nucleus over all layers and 0 is a background
        - cnt_extremes (CntExtremes object): where left, right, top, bottom attributes are coordinates
                                  of the corresponding extreme points of the biggest nucleus contour
    """
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
            biggest_nucleus_mask = utils.draw_cnts(nucleus_img.shape[:2], [current_nucleus_cnt])

    return biggest_nucleus_mask


def сut_out_mask(input_folder, output_folder, identifier, mask):
    """
    Cuts out an area that corresponds to the mask on each image (layer) located in the input_folder,
    saves processed images in the output_folder, and returns processed images combined into image_3d
    ---
        Parameters:
        - input_folder (string): path to the input folder with jpg images
        - output_folder (string): path to the folder to save processed jpg images
        - identifier (string) "actin" or "nucleus"
        - mask (np. array): stencil to cut out from the images
    ---
        Returns:
        - image_3d (np. array): three-dimensional array of processed (cut out) images combined layer by layer together

    """
    object_layers = []
    mask_cnt = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0][0]
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*_" + identifier + "_*.png"))):
        layer = int(img_path.rsplit(".", 1)[0].rsplit("_", 1)[1])
        object_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        object_layer = cv2.bitwise_and(object_img, mask)

        object_layers.append([object_layer, layer])

        cv2.imwrite(os.path.join(output_folder, os.path.basename(img_path)), object_layer)

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    image_3d = np.asarray([img for img, layer in object_layers], dtype=np.uint8)

    image_3d = np.moveaxis(image_3d, 0, -1)

    return image_3d


def get_yz_xsection(img_3d, output_folder, identifier, cnt_extremes):
    """
    Saves jpg images of yz cross-section of img_3d with padding to output_folder
    ---
        Parameters:
        - img_3d (np. array): the three-dimensional array that represents 3D image of the identifier
        - output_folder (string): path to the folder to save processed jpg images
        - identifier (string) "actin" or "nucleus"
        - cnt_extremes (CntExtremes object) where left, right, top, bottom attributes are coordinates
                                    of the corresponding extreme points of the biggest nucleus contour
    """
    top, bottom = cnt_extremes.top[1], cnt_extremes.bottom[1]
    x_start, x_end, step = cnt_extremes.left[0], cnt_extremes.right[0], 1

    for x_slice in range(x_start, x_end, step):
        xsection = img_3d[top: bottom, x_slice, :]
        img = xsection
        padded_img = utils.make_padding(img, UNET_IMG_SIZE[:2])

        img_path = os.path.join(output_folder, "xsection_" + identifier + "_" + str(x_slice) + ".png")
        cv2.imwrite(img_path, padded_img)


def get_3d_img(input_folder):
    """
    Reads and combines images from input_folder into image_3d according layer number.
    ---
        Parameters:
        - input_folder (string): path to the input folder with jpg images
    ---
        Returns:
        - image_3d (np.array): three-dimensional array of images combined layer by layer together
    """
    object_layers = []

    for img_path in glob.glob(input_folder + r"\*"):
        img_name, img_ext = os.path.splitext(os.path.basename(img_path))
        layer = int(img_name.rsplit("_", 1)[1]) #layer number is part of the image name

        img = cv2.imread(img_path, 0)
        img = np.flip(img, axis=0)
        object_layers.append([img, layer])

    object_layers = sorted(object_layers, key=lambda x: x[1], reverse=True)
    img_3d = np.asarray([mask for mask, layer in object_layers], dtype=np.uint8)

    return img_3d


def get_actin_cnt_objs(xsection, x):
    cnts = cv2.findContours(xsection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    actin_cnt_objs = []
    for cnt in cnts:
        z, y = utils.get_cnt_center(cnt)
        actin_cnt_objs.append(ActinContour(x, y, z, cnt))

    return actin_cnt_objs


def get_actin_fibers(img_3d):
    """
    Creates initial actin fibers based on the biggest intersection of actin contour on successive layers.

    Finds all contours of actin fibers on each yz layer and then add each of the contours to the existing actin fiber
    object if specified contour overlaps some contour from the previous layer. If the contour does not overlap any other
    contour from the previous layer, a new actin fiber object for this contour is created. If the contour overlaps more
    than one contours from the previous layer, the contour will be added to a fiber whose contour has the biggest
    overlap area. If two contours on the current layer overlap the same contour on the previous layer the contour with a
    bigger overlapping area will be added to the existed actin fiber object while a new actin fiber object would be
    created for the second contour.
    ---
        Parameters:
        - img_3d (np.array): A three-dimensional numpy array which represents a 3D image mask of actin fibers.
    ---
        Returns:
        - actin_fibers (List[ActinFaber]): List of ActinFiber objects
    """
    actin_fibers = []

    for x_slice in range(img_3d.shape[0]):
        print("Processing {} slice out of {} slices".format(x_slice, img_3d.shape[0]))

        xsection = img_3d[x_slice, :, :]

        actin_cnt_objs = get_actin_cnt_objs(xsection, x_slice)

        if x_slice == 0 or not actin_fibers:
            actin_fibers.extend([ActinFiber(actin_contour_obj.x,
                                            actin_contour_obj.y,
                                            actin_contour_obj.z,
                                            x_slice,
                                            actin_contour_obj.cnt)
                                 for actin_contour_obj in actin_cnt_objs])
        else:
            actin_fibers_from_previous_layer = [fiber for fiber in actin_fibers if fiber.last_layer[-1] == x_slice - 1]
            print("Number of actins from previous layer: {}".format(len(actin_fibers_from_previous_layer)))

            # find parents for all contours on new layer
            for new_layer_cnt_obj in actin_cnt_objs:
                for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                    new_layer_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(new_layer_cnt_mask, [new_layer_cnt_obj.cnt], -1, 255, -1)

                    actin_cnt_mask = np.zeros_like(xsection)
                    cv2.drawContours(actin_cnt_mask, [actin_fiber.cnts[-1]], -1, 255, -1)

                    intersection = np.count_nonzero(cv2.bitwise_and(new_layer_cnt_mask, actin_cnt_mask))
                    if intersection > 0 and intersection > new_layer_cnt_obj.xsection:
                        new_layer_cnt_obj.xsection = intersection
                        new_layer_cnt_obj.parent = i

            # assign contour to actin fibers from previous layer
            for i, actin_fiber in enumerate(actin_fibers_from_previous_layer):
                children_cnts = [new_layer_cnt for new_layer_cnt in actin_cnt_objs if new_layer_cnt.parent == i]

                if len(children_cnts) == 1:
                    new_layer_cnt = children_cnts[0]
                    actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)

                if len(children_cnts) > 1:
                    max_intersection, idx = 0, -1
                    for j, child_cnt in enumerate(children_cnts):
                        if child_cnt.xsection > max_intersection:
                            max_intersection = child_cnt.xsection
                            idx = j

                    new_layer_cnt = children_cnts[idx]
                    actin_fiber.update(new_layer_cnt.x, new_layer_cnt.y, new_layer_cnt.z, x_slice, new_layer_cnt.cnt)

                    for j, child_cnt in enumerate(children_cnts):
                        if j != idx:
                            actin_fibers.append(ActinFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))

            # create new ActinFibers for contour objects which were not assigned to any existed ActinFibers
            for child_cnt in actin_cnt_objs:
                if child_cnt.parent is None:
                    actin_fibers.append(ActinFiber(child_cnt.x, child_cnt.y, child_cnt.z, x_slice, child_cnt.cnt))

    return actin_fibers


def get_nucleus_origin(nucleus_3d_img):
    """
    Finds coordinate of nucleus anchor position which is:
    x is x of the center of the nucleus
    y is y of the center of the nucleus
    z is z the bottom of the nucleus
    ---
        Parameters:
        - nucleus_3d_img(np. array): the three-dimensional array that represents the 3D image of the nucleus
    ---
        Returns:
        - center_x, center_y, center_z (int, int, int) coordinates of the nucleus anchor position
    """
    center_x = nucleus_3d_img.shape[0] // 2

    slice_cnts = cv2.findContours(nucleus_3d_img[center_x, :, :], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]
    cnt_extremes = utils.get_cnt_extremes(slice_cnt)

    #center_y = cnt_extremes.bottom[1] - (cnt_extremes.bottom[1] - cnt_extremes.top[1]) // 2
    center_y = nucleus_3d_img.shape[1] // 2
    center_z = cnt_extremes.right[0]

    return center_x, center_y, center_z


def nucleus_reco_3d(nucleus_3d_img):
    points = []
    center_x, center_y, center_z = get_nucleus_origin(nucleus_3d_img)
    print(center_x, center_y, center_z)

    xdata, ydata, zdata = [], [], []
    volume = 0
    for slice in range(nucleus_3d_img.shape[0]):
        xsection_img = nucleus_3d_img[slice, :, :]

        slice_cnts = cv2.findContours(xsection_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if len(slice_cnts) != 0:
            slice_cnt = slice_cnts[np.argmax([len(cnt) for cnt in slice_cnts])]
            if IS_RESCALED:
                volume += cv2.contourArea(slice_cnt) * SCALE_X**3 #calculate volume based on counter area
                # volume += np.count_nonzero(utils.draw_cnts((512, 512), [slice_cnt]))* SCALE_X**3 #calculate volume based on pixel number

            else:
                volume += cv2.contourArea(slice_cnt) * SCALE_X * SCALE_Y * SCALE_Z #calculate volume based on counter area
                # volume += np.count_nonzero(utils.draw_cnts((512, 512), [slice_cnt]))* SCALE_X * SCALE_Y * SCALE_Z #calculate volume based on pixel number

            if slice % 15 == 0:
                ys = [pt[0, 0] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720] #720
                zs = [pt[0, 1] for idx, pt in enumerate(slice_cnt) if idx % 4 == 0 and pt[0, 0] < 720]

                xdata.extend([slice] * len(ys))
                ydata.extend(ys)
                zdata.extend(zs)

            cnt_ys = slice_cnt[:, 0, 1]
            cnt_zs = slice_cnt[:, 0, 0]

            points.extend([[x - center_x, y - center_y, center_z - z] for x, y, z in zip([slice]*len(cnt_ys), cnt_ys, cnt_zs)])

    np.savetxt("nucleus_points_coordinates.csv", np.array(points, dtype=int), delimiter=",", fmt="%10.0f")

    print("Nucleus volume: {}".format(volume))

    ax = plt.axes(projection='3d')
    ax.scatter3D(ydata, zdata, xdata, cmap='Greens', alpha=0.5)
    plt.show()

def save_mesh(image_3d):
    # idea was tacken here: https://forum.image.sc/t/3d-model-from-image-slices/33675/12
    p = 1
    z0 = np.zeros((p, image_3d.shape[1], image_3d.shape[2]), dtype=image_3d.dtype)
    z1 = np.zeros((image_3d.shape[0] + p * 2, p, image_3d.shape[2]), dtype=image_3d.dtype)
    image_3d = np.concatenate((z1, np.concatenate((z0, image_3d, z0), axis=0), z1), axis=1)

    verts, faces, normals, values = measure.marching_cubes(image_3d, 0, step_size=3)
    for v in verts:
        v[2] *= -1
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)

    surf_mesh_smooth = trimesh.Trimesh.smoothed(surf_mesh)
    surf_mesh.export('mesh_nuclei.stl')
    surf_mesh_smooth.export('mesh_nuclei_smooth.stl')
    print(f"The axis aligned box size of the nuclei: {surf_mesh.extents}")


def rescale_mask(input_folder, output_folder):
    """
    Rescale mask to produce 1x1x1 dimension image, so that sizes of each pixel
    on x,y,z dimentios are equal:
    """
    for img_path in tqdm(glob.glob(os.path.join(input_folder, "*.png"))):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized_img = utils.resize_mask(img, (SCALE_X, SCALE_Y, SCALE_Z), "width")
        img_name = os.path.basename(img_path)
        img_path_to_save = os.path.join(output_folder, img_name)
        cv2.imwrite(img_path_to_save, resized_img)


def main():
    # RUN_ACTIN_FIBERS_CREATION is a flag.
    # When RUN_ACTIN_FIBERS_CREATION is True algorithm runs from the beginning,
    # i.e starts from reading czi files(stack of microscopy images)
    # When RUN_ACTIN_FIBERS_CREATION is Facle, this algorithm analysis previously created object.
    RUN_ACTIN_FIBERS_CREATION = True

    if RUN_ACTIN_FIBERS_CREATION:
        print("\nRunning reader ...")
        output_folder = "single_nucleus_utils/temp/czi_layers"
        utils.prepare_folder(output_folder)
        start = time.time()
        reader = ConfocalImgReader(CONFOCAL_IMG, NOISE_TH, NUCLEUS_CHANNEL, ACTIN_CHANNEL)
        reader.read(output_folder)
        end = time.time() - start
        print(f"time is {end}")

        print("\nGenerate xsection images ...")
        input_folder = 'single_nucleus_utils/temp/czi_layers'
        output_folder = 'single_nucleus_utils/temp/actin_and_nucleus_layers'
        utils.prepare_folder(output_folder)
        biggest_nucleus_mask = find_biggest_nucleus_layer(input_folder)
        сut_out_mask(input_folder, output_folder, 'actin', biggest_nucleus_mask)
        сut_out_mask(input_folder, output_folder, 'nucleus', biggest_nucleus_mask)

        # rot_angle = - find_rotation_angle('single_nucleus_utils/temp/actin_and_nucleus_layers')
        rot_angle = -60  # TODO: automate later
        cnt_extremes = get_cnt_extremes(biggest_nucleus_mask, rot_angle)
        print(rot_angle)

        output_folder_actin = 'single_nucleus_utils/temp/actin_layers'
        utils.prepare_folder(output_folder_actin)
        input_folder = 'single_nucleus_utils/temp/actin_and_nucleus_layers'
        actin_3d_img = rotate_image(input_folder, "actin", rot_angle)
        get_yz_xsection(actin_3d_img, output_folder_actin, "actin", cnt_extremes)

        output_folder_nucleus = 'single_nucleus_utils/temp/nucleus_layers'
        utils.prepare_folder(output_folder_nucleus)
        nucleus_3d_img = rotate_image(input_folder, 'nucleus', rot_angle)
        get_yz_xsection(nucleus_3d_img, output_folder_nucleus, 'nucleus', cnt_extremes)

        print("\nRunning unet predictor for actin...")
        input_folder = output_folder_actin
        output_folder = 'single_nucleus_utils/temp/actin_mask'
        utils.prepare_folder(output_folder)
        run_predict_unet(input_folder, output_folder, ACTIN_UNET_MODEL, UNET_MODEL_SCALE, UNET_MODEL_THRESHOLD)

        print("\nRunning unet predictor for nucleus...")
        input_folder = output_folder_nucleus
        output_folder = 'single_nucleus_utils/temp/nucleus_mask'
        utils.prepare_folder(output_folder)
        run_predict_unet(input_folder, output_folder, NUCLEUS_UNET_MODEL, UNET_MODEL_SCALE, UNET_MODEL_THRESHOLD)
        utils.prepare_folder(r"single_nucleus_utils/temp/nucleus_mask_rescaled")
        utils.prepare_folder(r"single_nucleus_utils/temp/actin_mask_rescaled")
        if IS_RESCALED:
            rescale_mask(r"single_nucleus_utils/temp/actin_mask", r"single_nucleus_utils/temp/actin_mask_rescaled")
            rescale_mask(r"single_nucleus_utils/temp/nucleus_mask", r"single_nucleus_utils/temp/nucleus_mask_rescaled")

        print("\nFinding nuclei...")
        nucleus_3d_img = get_3d_img(r"single_nucleus_utils/temp/nucleus_mask_rescaled") if IS_RESCALED \
            else get_3d_img(r"single_nucleus_utils/temp/nucleus_mask")
        print(nucleus_3d_img.shape)
        nucleus_reco_3d(nucleus_3d_img)
        save_mesh(nucleus_3d_img)

        print("\nFinding actin fibers...")
        actin_img_3d = get_3d_img(r"single_nucleus_utils/temp/actin_mask_rescaled") if IS_RESCALED \
            else get_3d_img(r"single_nucleus_utils/temp/actin_mask")
        actin_fibers = get_actin_fibers(actin_img_3d)
        with open(ACTIN_OBJECT, "wb") as file_to_save:
            pickle.dump(actin_fibers, file_to_save)
    print("*******************************")
    actin_fibers = pickle.load(open(ACTIN_OBJECT, "rb"))
    get_actin_stat(actin_fibers, SCALE_X, SCALE_Y, SCALE_Z, IS_RESCALED)
    print("*******************************")
    # exit()

    print("\nRunning node creation and getting statistis data ...")
    run_node_creation(SCALE_X, SCALE_Y, SCALE_Z, ACTIN_OBJECT, IS_RESCALED, NEW_ACTIN_LEN_TH)


if __name__ == '__main__':
    main()
