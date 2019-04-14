import keras
from keras.models import load_model
from PIL import Image
import numpy as np
from utils import pre_process

# creating names to labels and inverse mapping.
labels_data = scipy.io.loadmat('./dataset/label_list.mat')
names_to_labels = {}
for k,i in enumerate(labels_data['label_list'][0]):
    names_to_labels.update({i[0]:k})
labels_to_names = {v: k for k, v in names_to_labels.items()}

# loading model
model = load_model('./snapshots/human_attribute_model_20.h5')
# load image
img_rgb  = Image.open('./dataset/photos/1005.jpg')
img_rgb = img_rgb.resize((100, 200))
img_batch = np.expand_dims(img_rgb, axis=0)
img_batch = pre_process(img_batch)
# prediction
results = model.predict(img_batch)
for i,score in enumerate(results[0]):
 if (score > 0.8):
    print(labels_to_names[i])