import glob
from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from keras.models import model_from_json
import os

PATH_TO_DATASET = str(Path.cwd().parent) + '/data/dataset_resampled/'
PATH_TO_MODELS = str(Path.cwd().parent) + '/models/'
IMAGE_SIZE=128

def read_train_images():
    '''
    Reads the training images stored locally.
    :return: numpy array of the training images
    '''
    images = []
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/train/*.jpg")]
    for i in range(0, len(filenames)):
        img_path = PATH_TO_DATASET + "images/train/img" + str(i) + ".jpg"
        images.append(resize(io.imread(img_path), (IMAGE_SIZE, IMAGE_SIZE, 3)))
    return np.array(images).astype('float32')


def read_validation_images():
    '''
    Reads the validation images stored locally.
    :return: numpy array of the validation images
    '''
    images = []
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/validation/*.jpg")]
    for i in range(0, len(filenames)):
        img_path = PATH_TO_DATASET + "images/validation/img" + str(i) + ".jpg"
        images.append(resize(io.imread(img_path), (IMAGE_SIZE, IMAGE_SIZE, 3)))
    return np.array(images).astype('float32')


def read_test_images():
    '''
     Reads the test images stored locally.
    :return: numpy array of the test images
    '''
    images = []
    filenames = [img for img in glob.glob(PATH_TO_DATASET + "images/test/*.jpg")]
    for i in range(0, len(filenames)):
        img_path = PATH_TO_DATASET + "images/test/img" + str(i) + ".jpg"
        images.append(resize(io.imread(img_path), (IMAGE_SIZE, IMAGE_SIZE, 3)))
    return np.array(images).astype('float32')


def read_train_labels():
    '''
    Reads the training labels stored locally.
    :return: numpy array containing the train labels
    '''
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_train.txt', dtype=int, delimiter=",")


def read_validation_labels():
    '''
    Reads the validation labels stored locally.
    :return:  numpy array containing the validation labels
    '''
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_validation.txt', dtype=int, delimiter=",")


def read_test_labels():
    '''
    Reads the test labels stored locally.
    :return:  numpy array containing the test labels
    '''
    return np.loadtxt(fname=PATH_TO_DATASET + 'labels/y_test.txt', dtype=int, delimiter=",")


def save_model(model, name):
    '''
    Saves the given model by the given name locally.
    :param model: The model to be saved
    :param name: The name by which the model should be saved
    '''
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
    '''
    Reads a model stored locally with the given name.
    :param name: Name of the model to be read
    :return: the loaded model
    '''
    # load json and create model
    json_file = open(PATH_TO_MODELS + name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(PATH_TO_MODELS + name + ".h5")
    print("Loaded model from disk")
    return loaded_model
