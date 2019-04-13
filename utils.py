import os 
import numpy as np

def get_img_ids():
	""" loading the img_ids.
	"""
	img_ids=[]
	for filename in os.listdir('./dataset/annotations/image-level'):
		img_ids.append(filename[:len(filename)-4])
	return img_ids

def pre_process(img_batch):
	""" Input data preprocessing.
	"""
	img_batch = img_batch.astype('float32')
	img_batch = img_batch / 255.0
	return img_batch

def one_hot_encode(integer_encodings, num_classes):
	""" One hot encode for multi-label classification.
	"""
	onehot_encoded = []
	for integer_encoded in integer_encodings:
		letter = [0 for _ in range(num_classes)]
		for value in integer_encoded:
			letter[value] = 1
		onehot_encoded.append(letter)

	onehot_encoded = np.array(onehot_encoded)
	return onehot_encoded