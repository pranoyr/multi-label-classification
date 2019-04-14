import keras
from keras.models import load_model
from PIL import Image
import numpy as np


# creating names to labels and inverse mapping.
labels_data = scipy.io.loadmat('/Users/pranoyr/PycharmProjects/multi-label-classification/dataset/label_list.mat')
names_to_labels = {}
for k,i in enumerate(labels_data['label_list'][0]):
    names_to_labels.update({i[0]:k})
labels_to_names = {v: k for k, v in names_to_labels.items()}



model = load_model('./snapshots/human_attribute_model_20.h5')
img_rgb  = Image.open('./dataset/photos/1005.jpg')
img_rgb = img_rgb.resize((100, 200))

img_batch = np.expand_dims(img_rgb, axis=0)


results = model.predict(img_batch)
idxs = np.argsort(results[0])[::-1][:2]
