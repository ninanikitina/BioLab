#PROJECT NAME
**Recognition of BioImages based on Machine Learning algorithms**


#DESCRIPTION

This project is solving the problem of detection nuclei and actin on a specific location within cells.
UNet model was trained from scratch with 300 images of 30 nuclei on different slices with data augmentation.


#CREDITS

We used Pytorch-UNet cloned from https://github.com/milesial/Pytorch-UNet by @milesial
*Note from @milesial: Use Python 3.6 or newer*
*Unet usage section was copied from the README file from this repository*

#USAGE

#Utils
- alternative_augumentation
	Augmentation based on the idea of catting nuclei from an image and then rotating the nuclei to a random angle
	The process of code wring for this augmentation is in progress.
- augumentaion_img
	Augmentation based on 
		- rotating an image to a random angle and then cutting out the central part
		- translating image by a random shifting of a window that equals to final rotated image size
czi_reader 
	Reads czi files and save jpg images form two different channels separately
	NOTE: initial jpgs are 16 bits, and this script converts it to 8 bits by using hardcoded normalization.
	The user should decide based on images preview how he would like to normalize an image
cut_nuclei
	Cuts a big image into a bunch of out 512"512 (hard codded size) images form with nuclei in the center and
	creates a mask based on contours
test_normalization

data_aug
	Pytorch augmentation that shown unsatisfactory results for this project.
	It will probably be used later.


#UNet

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`


## Prediction

After training your model and saving it to MODEL.pth, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```shell script
> python predict.py -h
usage: predict.py [-h] [--model FILE] --input INPUT [INPUT ...]
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [--mask-threshold MASK_THRESHOLD] [--scale SCALE]

Predict masks from input images

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 0.5)
```
You can specify which model file to use with `--model MODEL.pth`.

## Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 1

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
