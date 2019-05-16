import glob
from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from constants import IMAGE_SIZE
from keras.models import model_from_json
import os

PATH_TO_DATASET = str(Path.cwd().parent) + '/data/dataset/'
PATH_TO_MODELS = str(Path.cwd().parent) + '/models/'

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


def save_model(model, name):
    # serialize model to JSON
    if not os.path.exists(PATH_TO_MODELS):
        os.makedirs(PATH_TO_MODELS)
    model_json = model.to_json()
    with open(PATH_TO_MODELS + name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(PATH_TO_MODELS + name + ".h5")
    print("Saved model to disk")


def read_model(name):
    # load json and create model
    json_file = open(PATH_TO_MODELS + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(PATH_TO_MODELS + name + ".h5")
    print("Loaded model from disk")
