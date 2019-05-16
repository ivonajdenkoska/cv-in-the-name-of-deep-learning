import glob
from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from constants import IMAGE_SIZE

PATH_TO_DATASET = str(Path.cwd().parent) + '/data/dataset/'

def read_train_images():
    images = []
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/train/*.jpg")]
    for i in range(0, len(filenames)):
        img_path = PATH_TO_DATASET + "images/train/img" + str(i) + ".jpg"
        images.append(resize(io.imread(img_path), (IMAGE_SIZE, IMAGE_SIZE, 3)))
    return np.array(images)

def read_validation_images():
    images = []
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/validation/*.jpg")]
    for i in range(0, len(filenames)):
        img_path = PATH_TO_DATASET + "images/validation/img" + str(i) + ".jpg"
        images.append(resize(io.imread(img_path), (IMAGE_SIZE, IMAGE_SIZE, 3)))
    return np.array(images)

def read_train_labels():
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_train.txt', dtype=int, delimiter=",")

def read_validation_labels():
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_validation.txt', dtype=int, delimiter=",")
