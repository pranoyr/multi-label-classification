import os
import numpy as np
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import backend as K
from utils import get_img_ids
import random
from dataset import DataGenerator

# use this environment flag to change which GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Parameters
params = {'dim': (200, 100),
		  'batch_size': 32,
		  'n_classes': 59,
		  'n_channels': 3,
		  'shuffle': True}


# loading img_ids
img_ids = get_img_ids()

# train test split
random.shuffle(img_ids)
train_ids = img_ids[:800]
val_ids = img_ids[800:]

# Generators
training_generator = DataGenerator(train_ids, **params)
validation_generator = DataGenerator(val_ids, **params)

# Design model
# Create the model
# CONV => RELU => POOL
chanDim = -1
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(200,100,3)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# (CONV => RELU) * 2 => POOL
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (3, 3), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# use a *softmax* activation for single-label classification
# and *sigmoid* activation for multi-label classification
model.add(Dense(params['n_classes']))
model.add(Activation('sigmoid'))

# Compile model
# initialize the optimizer
EPOCHS = 75
INIT_LR = 1e-3
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# checkpoint
filepath="human_attribute_model_{epoch:02d}.h5"
checkpoint = ModelCheckpoint('./snapshots/'+filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train model on dataset
model.fit_generator(generator=training_generator,
					validation_data=validation_generator,
					epochs = EPOCHS,
					use_multiprocessing=True,
					workers=6,
					callbacks = callbacks_list)

# Score trained model.
scores = model.evaluate_generator(validation_generator)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])