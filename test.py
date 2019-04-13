import scipy.io
import os
import numpy as np
import cv2

# creating names to labels and inverse mapping.
labels_data = scipy.io.loadmat('/Users/pranoyr/PycharmProjects/multi-label-classification/dataset/label_list.mat')
names_to_labels = {}
for k,i in enumerate(labels_data['label_list'][0]):
    names_to_labels.update({i[0]:k})
labels_to_names = {v: k for k, v in names_to_labels.items()}

# create list 'labels' containing multi-label outputs and input array.
labels = []
inputs = []
for filename in os.listdir('./dataset/annotations/image-level'):
    mat = scipy.io.loadmat('./dataset/annotations/image-level/'+filename)
    inputs.append(cv2.imread('./photos/'+filename))
    labels.append(mat['tags'][0])
    print(mat['tags'][0])



def one_hot_encode(integer_encodings, num_classes):
    """ One hot encode for multi-label classification.
    """
    onehot_encoded = []
    for integer_encoded in integer_encodings:
        letter = [0 for _ in range(num_classes)]
        for value in integer_encoded:
            letter[value] = 1
        onehot_encoded.append(letter)
    return onehot_encoded






