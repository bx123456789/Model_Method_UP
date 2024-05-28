import cv2
import numpy as np
import random
from albumentations import Compose, Flip, Rotate, RandomResizedCrop, HueSaturationValue, Normalize
from albumentations.pytorch import ToTensorV2
from skimage import io
from .transform import CopyPaste, ISPRSLabelResize

# 数据增强管道
train_transform = Compose([
    CopyPaste(p=0.5),
    Flip(p=0.5),
    Rotate(30, p=0.5),
    RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0), p=0.5),
    HueSaturationValue(p=0.5),
    Normalize(),
    ToTensorV2(),
], additional_targets={'image_2': 'image'}, is_check_shapes=False)


# 读取图片路径并进行数据增强
def load_and_transform(image_A_path, image_B_path, label_path):
    imageA = io.imread(image_A_path)
    imageB = io.imread(image_B_path)
    label = cv2.imread(label_path, -1)

    transformed_data = train_transform(image=imageA, image_2=imageB, mask=label)
    imageA, imageB, label = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']

    return imageA, imageB, label


# 示例
image_A_path = 'path/to/imageA.png'
image_B_path = 'path/to/imageB.png'
label_path = 'path/to/label.png'

imageA, imageB, label = load_and_transform(image_A_path, image_B_path, label_path)

print(imageA.shape, imageB.shape, label.shape)
