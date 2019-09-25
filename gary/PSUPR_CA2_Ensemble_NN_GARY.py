# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 19:58:24 2019

@author: Cao Liang
"""

from gary.common_functions import *

from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
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

def resLyr(inputs, numFilters=16, kernelSz=3, strides=1, activation='relu',
           batchNorm=True, convFirst=True, lyrName=None):

    convLyr = Conv2D(numFilters, kernel_size=kernelSz, strides=strides, padding='same',
                     kernel_initializer='he_normal', kernel_regularizer=l2(1e-4), name=lyrName + '_conv' if lyrName else None)
    x = inputs
    if convFirst:
        x = convLyr(x)
        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
    else:
        if batchNorm:
            x = BatchNormalization(name=lyrName + '_bn' if lyrName else None)(x)
        if activation is not None:
            x = Activation(activation, name=lyrName + '_' + activation if lyrName else None)(x)
        x = convLyr(x)

    return x


def resBlkV1(inputs, numFilters=16, numBlocks=7, downsampleOnFirst=True, names=None):
    x = inputs
    for run in range(0, numBlocks):
        strides = 1
        blkStr = str(run + 1)
        if downsampleOnFirst and run == 0:
            strides = 2
        y = resLyr(inputs=x,
                   numFilters=numFilters,
                   strides=strides,
                   lyrName=names + '_Blk' + blkStr + '_Res1' if names else None)
        y = resLyr(inputs=y,
                   numFilters=numFilters,
                   activation=None,
                   lyrName=names + '_Blk' + blkStr + '_Res2' if names else None)
        if downsampleOnFirst and run == 0:
            x = resLyr(inputs=x,
                       numFilters=numFilters,
                       kernelSz=1,
                       strides=strides,
                       activation=None,
                       batchNorm=False,
                       lyrName=names + '_Blk' + blkStr + '_lin' if names else None)
        x = add([x, y],
                name=names + '_Blk' + blkStr + '_add' if names else None)
        x = Activation('relu',
                       name=names + '_Blk' + blkStr + '_relu' if names else None)(x)

    return x


def createResNetV1_7631(inputShape=(128, 128, 3), numClasses=3):
    inputs = Input(shape=inputShape)
    v = resLyr(inputs, lyrName='Input')
    v = Dropout(0.2)(v)

    v = resBlkV1(inputs=v, numFilters=16, numBlocks=7, downsampleOnFirst=False, names='Stg1')
    v = Dropout(0.2)(v)

    v = resBlkV1(inputs=v, numFilters=32, numBlocks=7, downsampleOnFirst=True, names='Stg2')
    v = Dropout(0.2)(v)

    v = resBlkV1(inputs=v, numFilters=64, numBlocks=7, downsampleOnFirst=True, names='Stg3')
    v = Dropout(0.2)(v)

    v = resBlkV1(inputs=v, numFilters=128, numBlocks=7, downsampleOnFirst=True, names='Stg4')
    v = Dropout(0.2)(v)

    v = AveragePooling2D(pool_size=8, name='AvgPool')(v)
    v = Dropout(0.2)(v)

    v = Flatten()(v)
    outputs = Dense(numClasses, activation='softmax', kernel_initializer='he_normal')(v)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=optmz, metrics=['accuracy'])

    return model


# load models from file
def load_all_models(n_models, model_name='PRMLS_CA2_Ensemble_NN',
                    model_files=[], models_creation_func=[]):
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        if len(model_files) == n_models:
            filename = model_files[i]
        else:
            filename = model_name + '_' + str(i + 1) + '.hdf5'

        # create model from model creation function list
        if len(models_creation_func) == n_models:
            model = models_creation_func[i]()
        else:
            model = createResNetV1_7631()

        # load model weight from file
        model.load_weights(filename)

        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models

# ----------------------------------
# All files pathes are defined below
# ----------------------------------
# Number of models
n_members = 2
# Ensemble model name
modelname = 'PRMLS_CA2_Ensemble_NN'
# Images data hdf5 file
img_data_h5_file = 'ca2data.h5'
# model hdf5 files
model_files_list = [modelname + '_1.hdf5', modelname + '_2.hdf5']
# model creation function list
model_create_list = [createResNetV1_7631, createResNetV1_7631]
# Ensemble model hdf5 file
filepath = modelname + ".hdf5"
# Ensemble model training data csv file
train_csv_path = modelname + '.csv'
# Ensemble model diagram pdf file
model_pdf_path = modelname + '1111_model.pdf'

# Set up 'ggplot' style
plt.style.use('ggplot')  # if want to use the default style, set 'classic'
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family'] = 'Arial'

X_train_data, y_train_data, X_test_data, y_test_data = read_data_set(h5_file=img_data_h5_file)

(trDat, trLbl) = X_train_data, y_train_data
(tsDat, tsLbl) = X_test_data, y_test_data

# Convert the data into 'float32'
# Rescale the values from 0~255 to 0~1
trDat = trDat.astype('float32') / 255
tsDat = tsDat.astype('float32') / 255

# Retrieve the row size of each image
# Retrieve the column size of each image
imgrows = trDat.shape[1]
imgclms = trDat.shape[2]
channel = trDat.shape[3]

# Perform one hot encoding on the labels
# Retrieve the number of classes in this problem
trLbl = to_categorical(trLbl)
tsLbl = to_categorical(tsLbl)
num_classes = tsLbl.shape[1]

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

optmz = optimizers.Adam(lr=0.0001)

# define the deep learning model
# load all models
members = load_all_models(n_members, model_files=model_files_list, models_creation_func=model_create_list)
print('Loaded %d models' % len(members))

stacked_model = define_stacked_model(members, numClasses=num_classes, model_optimizer=optmz)
print('Defined stacked model')
stacked_model.summary()

LRScheduler = LearningRateScheduler(lrSchedule)

# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest testing accuracy
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# Log the epoch detail into csv
csv_logger = CSVLogger(train_csv_path)
callbacks_list = [checkpoint, csv_logger, LRScheduler]

# Fit the model
# define ensemble model

print(len(members))

X_trDat = [trDat for model in members]
X_tsDat = [tsDat for model in members]

print("X_trDat: ", np.array(X_trDat).shape)
print("X_tsDat: ", np.array(X_tsDat).shape)

plot_model(stacked_model, to_file=model_pdf_path, show_shapes=True, show_layer_names=False, rankdir='TB')
'''
stacked_model.fit(X_trDat,
                  trLbl,
                  validation_data=(X_tsDat, tsLbl),
                  epochs=50,
                  verbose=1,
                  batch_size=64,
                  shuffle=True,
                  callbacks=callbacks_list)

# Now the training is complete, we get another object to load the weights
# compile it, so that we can do final evaluation on it
stacked_model_go = define_stacked_model(members, numClasses=num_classes, model_optimizer=optmz)
stacked_model_go.load_weights(filepath)

# Make classification on the test dataset
predicts = stacked_model_go.predict(X_tsDat)

# Prepare the classification output
# for the classification report
predout = np.argmax(predicts, axis=1)
testout = np.argmax(tsLbl, axis=1)
labelname = ['cat', 'bird', 'dog']

# the labels for the classfication report
testScores = metrics.accuracy_score(testout, predout)
confusion = metrics.confusion_matrix(testout, predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores * 100))
print(metrics.classification_report(testout, predout, target_names=labelname, digits=4))
print(confusion)

records = pd.read_csv(train_csv_path)
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0, 0.20, 0.40, 0.60, 0.80, 1.00])
plt.title('Loss value', fontsize=12)

ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0])
plt.title('Accuracy', fontsize=12)
plt.show()

plot_model(stacked_model, to_file=model_pdf_path, show_shapes=True, show_layer_names=False, rankdir='TB')
'''