import glob
import os


def prepare_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for f in glob.glob(output_folder + "/*"):
        os.remove(f)

