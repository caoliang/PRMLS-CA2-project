import h5py
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
import time
import datetime

def read_data_set(h5_file='out.h5'):
    with h5py.File(h5_file, 'r') as hf:
        X_train = hf['X_train'].value
        print('Read X_train: ', X_train.shape)

        y_train = hf['y_train'].value
        print('Read y_train: ', y_train.shape)

        X_test = hf['X_test'].value
        print('Read X_test: ', X_test.shape)

        y_test = hf['y_test'].value
        print('Read y_test: ', y_test.shape)

    return (X_train, y_train, X_test, y_test)

# Setup the models
def lrSchedule(epoch):
    lr = 1e-3

    if epoch > 160:
        lr *= 0.5e-3

    elif epoch > 120:
        lr *= 1e-3

    elif epoch > 80:
        lr *= 1e-2

    elif epoch > 40:
        lr *= 1e-1

    print('Learning rate: ', lr)

    return lr

'''
# define stacked model from multiple member input models
def define_stacked_model(members, numClasses=3,
                         out_model_img=None,
                         model_optimizer='adam'):
    # update all layers in all models to not be trainable
    for i in range(len(members)):
        model = members[i]
        for layer in model.layers:
            # rename to avoid 'unique layer name' issue
            # print(layer.__dict__)
            layer._name = 'ensemble_' + str(i + 1) + '_' + layer.name
            # make not trainable
            layer.trainable = False

    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    hidden = Dense(len(members) * numClasses, activation='relu')(merge)
    output = Dense(numClasses, activation='softmax')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)

    # plot graph of ensemble
    if out_model_img is not None:
        plot_model(model, to_file=out_model_img, show_shapes=True, show_layer_names=True)

    # compile
    model.compile(loss='categorical_crossentropy', optimizer=model_optimizer, metrics=['accuracy'])
    return model
'''

# Set up 'ggplot' style
plt.style.use('ggplot')     # if want to use the default style, set 'classic'
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'

X_train_data, y_train_data, X_test_data, y_test_data = read_data_set(h5_file='./../ca2data.h5' )
(trDat, trLbl) = X_train_data, y_train_data
(tsDat, tsLbl) = X_test_data, y_test_data

# Convert the data into 'float32'
# Rescale the values from 0~255 to 0~1
trDat = trDat.astype('float32')/255
tsDat = tsDat.astype('float32')/255

# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
trLbl = to_categorical(trLbl)
tsLbl = to_categorical(tsLbl)

num_classes = tsLbl.shape[1]

# fix random seed for reproducibility
seed = 29
np.random.seed(seed)

optmz = optimizers.Adam(lr=0.001)
modelname = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
print("modelname=========", modelname)