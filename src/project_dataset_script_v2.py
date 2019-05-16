from lxml import etree
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from pathlib import Path
from constants import IMAGE_SIZE

PATH_TO_ROOT = str(Path.cwd().parent)
PATH_TO_DATASET = PATH_TO_ROOT + '/data/dataset/'

# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = PATH_TO_ROOT + "/data/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball

# step1 - build list of filtered filenames
annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
annotation_files = os.listdir(annotation_folder)
filtered_filenames = []
for a_f in annotation_files:
    tree = etree.parse(os.path.join(annotation_folder, a_f))
    if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
        filtered_filenames.append(a_f[:-4])

# step2 - build (x,y) for TRAIN/VAL (classification)
classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
classes_files = os.listdir(classes_folder)
train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]


def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, IMAGE_SIZE, IMAGE_SIZE, 3) and  y np.ndarray of shape (n_images, n_classes)
    """
    temp = []
    train_labels = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
            label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
            train_labels.append(len(temp[-1]) * [label_id])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                       f in file]
    x = np.array([resize(io.imread(img_f), (IMAGE_SIZE, IMAGE_SIZE, 3)) for img_f in image_filenames]).astype(
        'float32')
    # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
    y_temp = []
    for tf in train_filter:
        y_temp.append([1 if tf in l else 0 for l in temp])
    y = np.array(y_temp)

    return x, y


x_train, y_train = build_classification_dataset(train_files)
print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
x_val, y_val = build_classification_dataset(val_files)
print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))

# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task
# (you will need a slightly different function for building the label images)

# Save images to disk
try:
    if not os.path.exists(PATH_TO_DATASET):
        os.makedirs(PATH_TO_DATASET)
    if not os.path.exists(PATH_TO_DATASET + 'images/train'):
        os.makedirs(PATH_TO_DATASET + 'images/train')
    if not os.path.exists(PATH_TO_DATASET + 'images/validation'):
        os.makedirs(PATH_TO_DATASET + 'images/validation')
    if not os.path.exists(PATH_TO_DATASET + 'labels'):
        os.makedirs(PATH_TO_DATASET + 'labels')
except OSError:
    print('Error: Creating directory of data')

for i, img in enumerate(x_train):
    name = PATH_TO_DATASET + 'images/train/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(x_val):
    name = PATH_TO_DATASET + 'images/validation/img' + str(i) + '.jpg'
    io.imsave(name, img)

# Save labels to disk
np.savetxt(PATH_TO_DATASET + "labels/y_train.txt", y_train)
np.savetxt(PATH_TO_DATASET + "labels/y_validation.txt", y_val)