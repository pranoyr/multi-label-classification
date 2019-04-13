import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from dataset import DataGenerator
from keras.optimizers import rmsprop
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from utils import get_img_ids
import random

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
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(200, 100, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(params['n_classes'], activation='sigmoid'))
# Compile model
epochs = 25
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

checkpoint = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    epochs = epochs,
                    use_multiprocessing=True,
                    workers=6,
                    callbacks = callbacks_list)
