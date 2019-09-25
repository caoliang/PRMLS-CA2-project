from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import MaxPooling2D

from common_functions import *

from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger,LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

def createSeqModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

model       = createSeqModel()  # This is meant for training
modelGo     = createSeqModel()  # This is used for final testing

model.summary()

LRScheduler     = LearningRateScheduler(lrSchedule)

# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest testing accuracy
filepath = modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# filepath        = modelname + ".hdf5"
# checkpoint      = ModelCheckpoint(filepath,
#                                   monitor='val_loss',
#                                   verbose=0,
#                                   save_best_only=True,
#                                   mode='min')
#
# # Log the epoch detail into csv
# csv_logger      = CSVLogger(modelname +'.csv')
# callbacks_list  = [checkpoint, csv_logger]

# Log the epoch detail into csv
csv_logger = CSVLogger(modelname +'.csv')
callbacks_list = [checkpoint, csv_logger, LRScheduler]

# Fit the model
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=20,
                             horizontal_flip=True,
                             vertical_flip=False)

# model.fit_generator(datagen.flow(trDat, trLbl, batch_size=32),
#           validation_data=(tsDat, tsLbl), epochs=10,
#           verbose=1, steps_per_epoch=len(trDat)/32)

model.fit(trDat,
          trLbl,
          validation_data=(tsDat, tsLbl),
          epochs=200,
          batch_size=64,
          shuffle=True,
          callbacks=callbacks_list)
# Now the training is complete, we get another object to load the weights
# compile it, so that we can do final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', optimizer=optmz, metrics=['accuracy'])

# Make classification on the test dataset
predicts = modelGo.predict(tsDat)

# Prepare the classification output for the classification report
predout = np.argmax(predicts,axis=1)
testout = np.argmax(tsLbl,axis=1)
labelname = ['cat', 'bird', 'dog']

testScores = metrics.accuracy_score(testout,predout)
confusion = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)

records = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0,0.20,0.40,0.60,0.80,1.00])
plt.title('Loss value',fontsize=12)

ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9,1.0])
plt.title('Accuracy',fontsize=12)
plt.show()

plot_model(model, to_file=modelname+'_model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


'''
LRScheduler     = LearningRateScheduler(lrSchedule)

# Create checkpoint for the training
# This checkpoint performs model saving when
# an epoch gives highest testing accuracy
filepath = modelname + ".hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

# Log the epoch detail into csv
csv_logger = CSVLogger(modelname +'.csv')
callbacks_list = [checkpoint,csv_logger,LRScheduler]

# Fit the model
datagen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             rotation_range=20,
                             horizontal_flip=True,
                             vertical_flip=False)

model.fit_generator(datagen.flow(trDat, trLbl, batch_size=32),
                    validation_data=(tsDat, tsLbl),
                    epochs=200, #originally 200
                    verbose=1,
                    steps_per_epoch=len(trDat)/32,
                    callbacks=callbacks_list)

# Now the training is complete, we get another object to load the weights
# compile it, so that we can do final evaluation on it
modelGo.load_weights(filepath)
modelGo.compile(loss='categorical_crossentropy', optimizer=optmz, metrics=['accuracy'])

# Make classification on the test dataset
predicts = modelGo.predict(tsDat)

# Prepare the classification output for the classification report
predout = np.argmax(predicts,axis=1)
testout = np.argmax(tsLbl,axis=1)
labelname = ['cat', 'bird', 'dog']

# the labels for the classfication report
testScores = metrics.accuracy_score(testout,predout)
confusion = metrics.confusion_matrix(testout,predout)

print("Best accuracy (on testing dataset): %.2f%%" % (testScores*100))
print(metrics.classification_report(testout,predout,target_names=labelname,digits=4))
print(confusion)

records     = pd.read_csv(modelname +'.csv')
plt.figure()
plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.yticks([0,0.20,0.40,0.60,0.80,1.00])
plt.title('Loss value',fontsize=12)

ax = plt.gca()
ax.set_xticklabels([])

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.yticks([0.6,0.7,0.8,0.9,1.0])
plt.title('Accuracy',fontsize=12)
plt.show()
'''