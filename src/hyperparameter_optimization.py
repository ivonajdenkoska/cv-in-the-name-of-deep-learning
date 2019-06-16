import pandas as pd
import talos as ta
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Model

from utils import *

def get_model(x_train, y_train, x_val, y_val, params):
    """
    Creates, compiles and trains model
    :return: Trained model and model's history object
    """
    input_img = Input(shape=(x, y, inChannel))
    conv1 = Conv2D(32, (3, 3), activation=params['activation'], padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation=params['activation'], padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation=params['activation'], padding='same')(pool2)

    drop1 = Dropout(rate=params['dropout'])(conv3)
    flat = Flatten()(drop1)
    den = Dense(128, activation=params['activation'])(flat)
    last = Dense(num_classes, activation=params['last_activation'])(den)

    full_model = Model(input_img, last)

    full_model.compile(optimizer=params['optimizer'],
                       loss=params['loss'],
                       metrics=['accuracy'])

    out = full_model.fit(x_train, y_train,
                         batch_size=params['batch_size'],
                         epochs=params['epochs'],
                         verbose=1,
                         validation_data=(x_val, y_val))

    return out, full_model

# Read train and validation data
x_train = read_train_images()
x_val = read_validation_images()

y_train = read_train_labels()
y_val = read_validation_labels()

inChannel = x_train.shape[3]
x, y = x_train.shape[1], x_train.shape[2]
num_classes = len(y_train[0])

# specify parameters and distributions to sample from
p={'batch_size': [8, 16, 32],
   'epochs': [25, 50],
   'dropout': [0.1, 0.2],
   'optimizer': ['Adam', 'RMSprop', 'SGD'],
   'loss': ['hinge', 'binary_crossentropy'],
   'activation':['relu', 'elu'],
   'last_activation': ['sigmoid','softmax']}

# Perform the random search for hyperparameter optimization
h = ta.Scan(x_train, y_train,
            params=p,
            model=get_model,
            dataset_name='rand_search',
            experiment_no='1',
            grid_downsample=0.05)

# Accessing the results data frame
h = pd.read_csv('rand_search_1.csv', delimiter = ',')

r = ta.Reporting('rand_search_1.csv')
