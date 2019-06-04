import os
import shutil
from pathlib import Path

PATH_TO_ROOT = str(Path.cwd().parent)
PATH_TO_DATASET = PATH_TO_ROOT + '/data/dataset/'
PATH_TO_SEGMENTATION = PATH_TO_DATASET + 'segmentation/'
PATH_TO_AUGMENTED_IMAGES = PATH_TO_SEGMENTATION + 'augmented_images/'

# Check if the directory already exists
if os.path.exists(PATH_TO_DATASET):
    shutil.rmtree(PATH_TO_DATASET)  # Delete it if it exists

# Create folders
try:
    os.makedirs(PATH_TO_DATASET)
    os.makedirs(PATH_TO_DATASET + 'images/train')
    os.makedirs(PATH_TO_DATASET + 'images/validation')
    os.makedirs(PATH_TO_DATASET + 'images/test')
    os.makedirs(PATH_TO_DATASET + 'labels')

    os.makedirs(PATH_TO_SEGMENTATION)
    os.makedirs(PATH_TO_SEGMENTATION + 'train')
    os.makedirs(PATH_TO_SEGMENTATION + 'validation')
    os.makedirs(PATH_TO_SEGMENTATION + 'test')
    os.makedirs(PATH_TO_SEGMENTATION + 'train-labels')
    os.makedirs(PATH_TO_SEGMENTATION + 'validation-labels')
    os.makedirs(PATH_TO_SEGMENTATION + 'test-labels')
    os.makedirs(PATH_TO_SEGMENTATION + 'processed_labels')

    os.makedirs(PATH_TO_AUGMENTED_IMAGES)
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'train')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'validation')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'test')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'train-labels')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'validation-labels')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'test-labels')
    os.makedirs(PATH_TO_AUGMENTED_IMAGES + 'processed_labels')
except OSError:
    print('Error: Creating directory of data')