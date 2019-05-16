from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import os

from utils import *

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0" #model will be trained on GPU 1

x_train = read_train_images()
x_val = read_validation_images()
y_train = read_train_labels()
y_val = read_validation_labels()

batch_size = 64
epochs = 20
inChannel = 3
x, y = IMAGE_SIZE, IMAGE_SIZE
input_img = Input(shape = (x, y, inChannel))

def autoencoder(input_img):
    #encoder

    conv1 = Conv2D(32, (3, 3), data_format="channels_last", activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), data_format="channels_last", activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding='same')(pool2)

    #decoder
    conv4 = Conv2D(128, (3, 3), data_format="channels_last", activation='relu', padding='same')(conv3)
    up1 = UpSampling2D((2,2))(conv4) # 14 x 14 x 128
    conv5 = Conv2D(64, (3, 3), data_format="channels_last", activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2,2))(conv5) # 28 x 28 x 64
    decoded = Conv2D(3, (3, 3), data_format="channels_last", activation='sigmoid', padding='same')(up2)
    return decoded


autoencoder = Model(input_img, autoencoder(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

# Train the model
autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,
                                    verbose=1,validation_data=(x_val, x_val))

loss = autoencoder_train.history['loss']

val_loss = autoencoder_train.history['val_loss']
epochs = range(epochs)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

save_model(autoencoder, "autoencoder_color")
