from torchvision import transforms as tf
import torchvision.transforms.functional as TF
import random


class DataAugmentation:
    def __init__(self):
        pass

    @staticmethod
    def get_data_augmentation_transforms():
        transforms = tf.Compose([
            tf.RandomHorizontalFlip(),
            tf.RandomVerticalFlip(),
            #tf.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.2)),
            RotationTransform(angles=[0, 90, 180, 270])
        ])

        return transforms


class RotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)