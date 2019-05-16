import glob
from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from constants import IMAGE_SIZE

PATH_TO_DATASET = str(Path.cwd().parent) + '/data/dataset/'

def read_train_images():
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/train/*.jpg")]
    filenames.sort()
    return np.array([resize(io.imread(img_f), (IMAGE_SIZE, IMAGE_SIZE, 3)) for img_f in filenames]).astype('float32')

def read_validation_images():
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/validation/*.jpg")]
    filenames.sort()
    return np.array([resize(io.imread(img_f), (IMAGE_SIZE, IMAGE_SIZE, 3)) for img_f in filenames]).astype('float32')

def read_train_labels():
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_train.txt', dtype=int, delimiter=",")

def read_validation_labels():
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_validation.txt', dtype=int, delimiter=",")
