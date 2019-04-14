import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from utils import pre_process
import scipy.io

# creating names to labels and inverse mapping.
labels_data = scipy.io.loadmat('./dataset/label_list.mat')
names_to_labels = {}
for k,i in enumerate(labels_data['label_list'][0]):
    names_to_labels.update({i[0]:k})
labels_to_names = {v: k for k, v in names_to_labels.items()}



model = load_model('./snapshots/human_attribute_model_20.h5')
img_rgb  = Image.open('./sample.jpg')
img_rgb = img_rgb.resize((100, 200))

img_batch = np.expand_dims(img_rgb, axis=0)
img_batch = pre_process(img_batch)

results = model.predict(img_batch)
for i,score in enumerate(results[0]):
 if (score > 0.7):
    print(labels_to_names[i])
