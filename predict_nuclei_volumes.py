import os
import glob

from multiple_nuclei_utils.czi_reader import run_czi_reader
from multiple_nuclei_utils.cut_nuclei import run_cut_nuclei
from multiple_nuclei_utils.reconstruct_layers import merge_individual_masks
from multiple_nuclei_utils.volume_estimation import run_volume_estimation
from alexnet.predict import run_predict_alexnet
from unet.predict import run_predict_unet


CZI_IMG = r"D:\BioLab\img\3Dimg_for_tests\3D\20x\control_siRNA_20x_5x5_15um_stack_Airyscan_Processing_Stitch-003.czi"
NUCLEI_CHANNEL = 1
NOISE_TH = 0.0001
ALEXNET_MODEL = r"D:\BioLab\models\CP_epoch150_with_new_data_aug_where_less_0.pth"
ALEXNET_DICISION_THRESHOLD = 0.5
UNET_MODEL = r"D:\BioLab\models\CP_epoch15_unet_no_data-aug.pth"
UNET_MODEL_SCALE = 0.5
UNET_MODEL_THRESHOLD = 0.5
SCALE_X = 0.25
SCALE_Y = 0.25
SCALE_Z = 0.5


if __name__ == '__main__':
    print("\nRunning czi reader ...")
    h, w, layers = run_czi_reader(CZI_IMG, NUCLEI_CHANNEL, NOISE_TH)
    # h, w, layers = 8444, 8438, 31

    print("\nRunning nuclei cutter ...")
    run_cut_nuclei()

    print("\nRunning alexnet predictor ...")
    run_predict_alexnet(None, None, ALEXNET_MODEL, ALEXNET_DICISION_THRESHOLD, False)

    print("\nRunning unet predictor ...")
    run_predict_unet(None, None, UNET_MODEL, UNET_MODEL_SCALE, UNET_MODEL_THRESHOLD)

    print("\nMerging individual masks into one layer ...")
    merge_individual_masks(None, None, layers, h, w)

    print("\nRunning volume estimation ...")
    run_volume_estimation(None, None, SCALE_X, SCALE_Y, SCALE_Z)
