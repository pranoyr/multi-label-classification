import numpy as np
import keras
import os
import scipy.io
from PIL import Image
from utils import *


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, batch_size=32, dim=(32, 32, 23), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.n = len(list_IDs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = []
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img_rgb = Image.open('./dataset/photos/'+ID+'.jpg')
            img_rgb = img_rgb.resize((self.dim[1], self.dim[0]))
            X[i, ] = img_rgb
            # Store class
            mat = scipy.io.loadmat('./dataset/annotations/image-level/'+ID+'.mat')
            y.append(mat['tags'][0])

        # preprocessing data
        X = pre_process(X)
        return X, one_hot_encode(y, num_classes=self.n_classes)
