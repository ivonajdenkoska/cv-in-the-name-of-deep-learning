import cv2
import numpy as np
from skimage import io
from lxml import etree
from mxnet import image, nd
from skimage.transform import resize

from constants import IMAGE_SIZE
from sklearn.model_selection import train_test_split

from create_folders import *

# parameters that you should set before running this script
filter = ['aeroplane', 'cat']   # select class, this default should yield 1489 training and 1470 validation images
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


### CLASSIFICATION
def build_classification_dataset(list_of_files):
    """ build training or validation set

    :param list_of_files: list of filenames to build trainset with
    :return: tuple with x np.ndarray of shape (n_images, IMAGE_SIZE, IMAGE_SIZE, 3)
             and  y np.ndarray of shape (n_images, n_classes)
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
val_images, val_labels = build_classification_dataset(val_files)
x_val, x_test, y_val, y_test = train_test_split(val_images, val_labels, test_size=0.2, random_state=42)

print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))
print('%i test images from %i classes' %(x_test.shape[0],  y_test.shape[1]))


# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task
# (you will need a slightly different function for building the label images)

for i, img in enumerate(x_train):
    name = PATH_TO_DATASET + 'images/train/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(x_val):
    name = PATH_TO_DATASET + 'images/validation/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(x_test):
    name = PATH_TO_DATASET + 'images/test/img' + str(i) + '.jpg'
    io.imsave(name, img)

# Save labels to disk
np.savetxt(PATH_TO_DATASET + "labels/y_train.txt", y_train.astype(int), fmt='%i', delimiter=",")
np.savetxt(PATH_TO_DATASET + "labels/y_validation.txt", y_val.astype(int), fmt='%i', delimiter=",")
np.savetxt(PATH_TO_DATASET + "labels/y_test.txt", y_test.astype(int), fmt='%i', delimiter=",")

### SEGMENTATION
filter.insert(0, "background")
n_classes = len(filter)

# Use this as a lookup for the colors and the classes
VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

indexes = [VOC_CLASSES.index(filter_item) for filter_item in filter]
VOC_COLORMAP_FILTER = [VOC_COLORMAP[index] for index in indexes]


def build_segmentation_dataset(list_of_files):
    """ build training or validation set for the segmentation task

    :param list_of_files: list of filenames to build dataset with
    :return: tuple with features np.ndarray of shape (n_images, IMAGE_SIZE, IMAGE_SIZE, 3) and
    labels np.ndarray of shape (n_images, IMAGE_SIZE, IMAGE_SIZE, 3)
    """
    temp = []
    for f_cf in list_of_files:
        with open(f_cf) as file:
            lines = file.read().splitlines()
            temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
    train_filter = [item for l in temp for item in l]

    image_folder = os.path.join(voc_root_folder, "VOC2009/SegmentationClass/")
    segmentation_filter = [f for f in train_filter for file in os.listdir(image_folder) if f in file]

    features, labels = [None] * len(segmentation_filter), [None] * len(segmentation_filter)

    image_filenames = [os.path.join(image_folder, file) for f in segmentation_filter
                       for file in os.listdir(image_folder) if f in file]

    for i, img_filename in enumerate(image_filenames):
        labels[i] = image.imread(img_filename)

    image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
    image_filenames = [os.path.join(image_folder, file) for f in segmentation_filter
                       for file in os.listdir(image_folder) if f in file]

    for i, img_filename in enumerate(image_filenames):
        features[i] = image.imread(img_filename)

    return features, labels


def convert_and_resize(data):
    '''
    Convert the data into numpy array and resize the images to (IMAGE_SIZE, IMAGE_SIZE)
    :param data: The data containing the images to be resized
    :return: numpy array with resized images
    '''
    converted_data = []
    for image in data:
        img = image.asnumpy()
        img = resize(img, (IMAGE_SIZE, IMAGE_SIZE)).astype('float32')
        converted_data.append(img)
    return np.array(converted_data)


def voc_label_indices(colormap, colormap2label):
    '''
    Convert an RGB image to image with labels (numbers) for each color
    :param colormap: a segmented image where every class is represented by different color
    :param colormap2label: NDArray containing the label for each color
    :return: a signle matrix containing the labels for each pixel in the image
    '''
    colormap = colormap.astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256 + colormap[:, :, 2])
    return colormap2label[idx]


def get_segmentation_array(img, nClasses, width, height):
    '''
    Convert an RGB image with shape [width, height, 3] to nClasses binary matrices
    :param img: a segmented image
    :param nClasses: number of classes
    :param width: width of the image
    :param height: height of the image
    :return: binary matrices for every class
    '''
    seg_labels = np.zeros((height, width , nClasses))
    img = cv2.resize(img, (width, height))
    img = img[:, :, 0]
    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)
    return seg_labels


def get_segmentation_labels(segmentations):
    '''
    For each segmented image create binary matrices for each class
    :param segmentations: list of segmentations
    :return: labels for each segmented image that contain binary matrices for every class
    '''
    seg_labels = []
    for image in segmentations:
        img = voc_label_indices(image, colormap2label)
        img_stacked = np.dstack([img.asnumpy(), img.asnumpy(), img.asnumpy()])
        img_mod = get_segmentation_array(img_stacked, n_classes, IMAGE_SIZE, IMAGE_SIZE)
        seg_labels.append(img_mod)
    return seg_labels


train_features_raw, train_segmentations_raw = build_segmentation_dataset(train_files)
validation_features_raw, validation_segmentations_raw = build_segmentation_dataset(val_files)
# Split validation set into validation and test
validation_features_raw, test_features_raw, validation_segmentations_raw, test_segmentations_raw = \
    train_test_split(validation_features_raw, validation_segmentations_raw, test_size=0.2, random_state=42)

print(len(train_features_raw))
print(len(validation_features_raw))
print(len(test_features_raw))

train_features_resized = convert_and_resize(train_features_raw)
train_segmentations_resized = convert_and_resize(train_segmentations_raw)
validation_features_resized = convert_and_resize(validation_features_raw)
validation_segmentations_resized = convert_and_resize(validation_segmentations_raw)
test_features_resized = convert_and_resize(test_features_raw)
test_segmentations_resized = convert_and_resize(test_segmentations_raw)


# Save resized images to disk
for i, img in enumerate(train_features_resized):
    name = PATH_TO_SEGMENTATION + 'train/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(validation_features_resized):
    name = PATH_TO_SEGMENTATION + 'validation/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(test_features_resized):
    name = PATH_TO_SEGMENTATION + 'test/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(train_segmentations_resized):
    name = PATH_TO_SEGMENTATION + 'train-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(validation_segmentations_resized):
    name = PATH_TO_SEGMENTATION + 'validation-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(test_segmentations_resized):
    name = PATH_TO_SEGMENTATION + 'test-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

# Process the labels
colormap2label = nd.zeros(256 ** 3)
for i, colormap in enumerate(VOC_COLORMAP_FILTER):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i

train_seg_labels = get_segmentation_labels(train_segmentations_raw)
val_seg_labels = get_segmentation_labels(validation_segmentations_raw)
test_seg_labels = get_segmentation_labels(test_segmentations_raw)

outfile_train = PATH_TO_SEGMENTATION + 'processed_labels/train.npy'
np.save(outfile_train, train_seg_labels)
outfile_validation = PATH_TO_SEGMENTATION + 'processed_labels/validation.npy'
np.save(outfile_validation, val_seg_labels)
outfile_test = PATH_TO_SEGMENTATION + 'processed_labels/test.npy'
np.save(outfile_test, test_seg_labels)

print("Segmentation images and labels saved to disk")

# AUGMENTED IMAGES
def random_shift_scale_rotate(image, mask, label, angle, scale, aspect, shift_dx, shift_dy,
                              borderMode=cv2.BORDER_CONSTANT, u=0.5):
    '''
    Perform random augmentations on the given image, mask and label
    :param image: the image for which to generate augmentations
    :param mask: the segmented image
    :param label: the binary matrices
    :param angle: the angle by which to rotate the image
    :param scale: the scale
    :param aspect: aspect ratio
    :param shift_dx: how much to shift the image in x direction
    :param shift_dy: how much to shift the image in y direction
    :param borderMode: pixel extrapolation method
    :param u: a number
    :return: randomly shifted, scaled and rotated image, mask and label
    '''
    if np.random.random() < u:
        height, width, channels = image.shape

        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(shift_dx * width)
        dy = round(shift_dy * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(1, 1, 1, 1))

        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR,
                                   borderMode=borderMode, borderValue=(1, 1, 1, 1))

        label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR,
                                    borderMode=borderMode, borderValue=(1, 1, 1, 1))
    return image, mask, label


def random_horizontal_flip(image, mask, label, u=0.5):
    '''
    Perform a random horizontal flip
    :param image: the original image
    :param mask: the segmented image
    :param label: the binary matrices
    :param u: a number
    :return: horizontally flipped image, mask and label
    '''
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        label = cv2.flip(label, 1)

    return image, mask, label


def augment_img(img, mask, label):
    '''
    Augment the given image
    :param img: the image to be augmented
    :param mask: the corresponding segmented image
    :param label: the corresponding binary matrices
    :return: augmented image, mask and label
    '''
    rotate_limit = (-45, 45)
    aspect_limit = (0, 0)
    scale_limit = (-0.1, 0.1)
    shift_limit = (-0.0625, 0.0625)
    shift_dx = np.random.uniform(shift_limit[0], shift_limit[1])
    shift_dy = np.random.uniform(shift_limit[0], shift_limit[1])
    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])

    img, mask, label = random_shift_scale_rotate(img, mask, label, angle, scale, aspect, shift_dx, shift_dy)
    img, mask, label = random_horizontal_flip(img, mask, label)
    return img, mask, label


def augment_images(features, segmentations, labels):
    '''
    Generate 5 augmented  images for each image in the list
    :param features: the original images
    :param segmentations: the corresponding segmentations
    :param labels: the corresponding binary matrices
    :return: list of augmented images, segmentations and labels
    '''
    augmented_features, augmented_segmentations, augmented_labels = [], [], []
    for feature, segmentation, label in zip(features, segmentations, labels):
        for i in range(5):
            img, mask, label = augment_img(feature, segmentation, label)
            augmented_features.append(img)
            augmented_segmentations.append(mask)
            augmented_labels.append(label)
    return augmented_features, augmented_segmentations, augmented_labels


train_augmented_features, train_augmented_segmentations, train_seg_labels_augmented = augment_images(
    train_features_resized, train_segmentations_resized, train_seg_labels)
validation_augmented_features, validation_augmented_segmentations, val_seg_labels_augmented = augment_images(
    validation_features_resized, validation_segmentations_resized, val_seg_labels)
test_augmented_features, test_augmented_segmentations, test_seg_labels_augmented = augment_images(
    test_features_resized, test_segmentations_resized, test_seg_labels)

for i, img in enumerate(train_augmented_features):
    name = PATH_TO_AUGMENTED_IMAGES + 'train/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(validation_augmented_features):
    name = PATH_TO_AUGMENTED_IMAGES + 'validation/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(test_augmented_features):
    name = PATH_TO_AUGMENTED_IMAGES + 'test/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(train_augmented_segmentations):
    name = PATH_TO_AUGMENTED_IMAGES + 'train-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(validation_augmented_segmentations):
    name = PATH_TO_AUGMENTED_IMAGES + 'validation-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

for i, img in enumerate(test_augmented_segmentations):
    name = PATH_TO_AUGMENTED_IMAGES + 'test-labels/img' + str(i) + '.jpg'
    io.imsave(name, img)

outfile_train = PATH_TO_AUGMENTED_IMAGES + 'processed_labels/train.npy'
np.save(outfile_train, train_seg_labels_augmented)
outfile_validation = PATH_TO_AUGMENTED_IMAGES + 'processed_labels/validation.npy'
np.save(outfile_validation, val_seg_labels_augmented)
outfile_test = PATH_TO_AUGMENTED_IMAGES + '/processed_labels/test.npy'
np.save(outfile_test, test_seg_labels_augmented)

print("Augmented images and labels saved to disk")
