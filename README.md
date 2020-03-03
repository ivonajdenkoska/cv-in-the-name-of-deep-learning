# In the Name of Deep Learning

The aim of this project is to explore the application of Deep Learning in the field of Computer Vision, as part of the [Computer Vision course](https://onderwijsaanbod.kuleuven.be/2019/syllabi/e/H02A5AE.htm#activetab=doelstellingen_idm18484000) @ KU Leuven, for the academic year 2018/2019.

The project work is divided in three main sections: Autoencoders, Classification and Segmentation. The whole implementation of the project is done in Python using the following libraries: Keras, Numpy, Pandas, Scikit Learn and Talos.

The dataset used for this project is a subset of the famous PASCAL VOC-2009, which is well-known as standardized image dataset for object class recognition.

## Autoencoder
An autoencoder is an unsupervised neural network, with a specific task to learn how to compress the input in such way that it minimizes the reconstruction error. In the context of image classification, autoencoders can be used as feature extractors, which can be fed afterwards to another model for classification.

The input size for all autoencoders in this project is 128x128x3, representing the height, width and depth of the input image respectively. We tested three autoencoder settings with different coding variables. For the first autoencoder the input is compressed to 16x16x32, or 8192 coding variables, the second autoencoder: 8x8x16 or 1024 coding variables and the third autoencoder: 4x4x8 or 128 coding variables. 

## Image Classification
The neural network is formed by taking the encoder part from the previously built convolutional autoencoder and adding a fully–connected layer, which outputs the class of the given image. Here, the encoder part has the role of feature extractor. The output connections from the encoder are flattened and then fed into dense layer which uses sigmoid activation function. We use three approaches for the initialization of the weights:
  - **Freeze and pretrain**: Freeze the pretrained layers from the convolutional autoencoder, and train the fully–connected part. Freezing the layers from the autoencoder means that the weights do not change during backpropagation and only the last classification layer is trained.
  - **Fine-tune**: The weights from the first approach are loaded into the model and the complete network is trained again, without freezing any layers. In this way, we fine-tune the weights of the pretrained part.
  - **Randomly initialized**: Train from scratch all the weights of the network, while keeping the same structure. 

## Image Segmentation
For this task, we chose to focus on category-level segmentation. Therefore, we used the images given in the JPEGImages folder as input, and the labels (ground truth segmentations) given in the SegmentationClass folder. We consider two approaches: 
  - **FCN-32s** to predict the class of each of the pixels (aeroplane, cat or background).
  Fully convolutional networks are composed of convolutional layers without any fully-connected layers. The output of the network has the same spatial size as the input image (width and height) with a channel depth equivalent to the number of possible classes to be predicted.
  - **U-Net** to predict each pixel as foreground or background (binary classification per pixel). 
  The first part of this network may be thought of as an encoder where convolution and maxpooling operations are applied to encode the input image into feature representations at multiple levels. The second part corresponds to a decoder that performs upsampling and regular convolution operations.
 
The performance of the networks is measured using Dice Score.
