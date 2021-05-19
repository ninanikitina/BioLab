import argparse
import logging
import os
import glob

import numpy as np
import torch
import torchvision.models.alexnet as alexnet
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def predict_img(net,
                full_img,
                device,
                decision_threshold=0.5):
    net.eval()

    data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.ToTensor(),
                                          # tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                  ])

    img = data_transforms(full_img)

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        probs = torch.sigmoid(output)

        probs = probs.squeeze(0).cpu().item()

    return probs, int(probs > decision_threshold)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='Path to folder with input images', required=True)
    parser.add_argument('--output', '-o', metavar='INPUT',
                        help='Path to folder with ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--decision-threshold', '-t', type=float,
                        help="Minimum probability value to consider that nucleus on the image is real",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def run_predict_alexnet(folder_path, output_folder_path, model_path, decision_threshold, save_non_nuclei=False):
    if folder_path is None:
        if not os.path.exists('temp/nucleus_imgs'):
            raise RuntimeError("There is no folder {}\nCan't process images".format("temp/nucleus_imgs"))

        if not os.path.exists('temp/true_nucleus_imgs'):
            os.makedirs('temp/true_nucleus_imgs')

        if save_non_nuclei and not os.path.exists('temp/false_nucleus_imgs'):
            os.makedirs('temp/false_nucleus_imgs')

        folder_path = 'temp/nucleus_imgs'
        output_folder_path = 'temp/true_nucleus_imgs'
        non_nucleus_output_folder_path = 'temp/false_nucleus_imgs'

    net = alexnet(num_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")
    for img_path in tqdm(glob.glob(os.path.join(folder_path, "*.png"))):
        logging.info("\nPredicting image {} ...".format(img_path))
        img = Image.open(img_path).convert('RGB')

        prob, label_pred = predict_img(net=net,
                                       full_img=img,
                                       decision_threshold=decision_threshold,
                                       device=device)

        if label_pred == 1:
            img_name = os.path.basename(img_path)
            img_path_to_save = os.path.join(output_folder_path, img_name)
            img.save(img_path_to_save)
        elif save_non_nuclei:
            img_name = os.path.basename(img_path)
            img_path_to_save = os.path.join(non_nucleus_output_folder_path, img_name)
            img.save(img_path_to_save)

        logging.info("\nProbabilty of true nucleus {}".format(prob))


if __name__ == "__main__":
    args = get_args()
    out_files = get_output_filenames(args)

    run_predict_alexnet(args.input, args.output, args.model, args.decision_threshold, save_non_nuclei=True)
