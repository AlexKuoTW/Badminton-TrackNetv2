import numpy as np
import sys, getopt
import os
from glob import glob
import piexif
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import array_to_img, img_to_array, load_img
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.layers import *
from TrackNet3 import TrackNet3
import keras.backend as K
from keras import optimizers
from keras.activations import *
import tensorflow as tf
import cv2
import math
import matplotlib.pyplot as plt

import math
import tensorflow as tf
from sklearn.utils.extmath import cartesian

BATCH_SIZE=4
HEIGHT=288
WIDTH=512
#HEIGHT=360
#WIDTH=640
mag = 1
sigma = 2.5

#Return the numbers of true positive, true negative, false positive and false negative
def outcome(y_pred, y_true, tol):
	n = y_pred.shape[0]
	i = 0
	TP = TN = FP1 = FP2 = FN = 0
	while i < n:
		for j in range(3): #1
			if np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) == 0:
				TN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) == 0:
				FP2 += 1
			elif np.amax(y_pred[i][j]) == 0 and np.amax(y_true[i][j]) > 0:
				FN += 1
			elif np.amax(y_pred[i][j]) > 0 and np.amax(y_true[i][j]) > 0:
				h_pred = y_pred[i][j] * 255
				h_true = y_true[i][j] * 255
				h_pred = h_pred.astype('uint8')
				h_true = h_true.astype('uint8')
				#h_pred
				(cnts, _) = cv2.findContours(h_pred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_pred, cy_pred) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))

				#h_true
				(cnts, _) = cv2.findContours(h_true.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
				rects = [cv2.boundingRect(ctr) for ctr in cnts]
				max_area_idx = 0
				max_area = rects[max_area_idx][2] * rects[max_area_idx][3]
				for j in range(len(rects)):
					area = rects[j][2] * rects[j][3]
					if area > max_area:
						max_area_idx = j
						max_area = area
				target = rects[max_area_idx]
				(cx_true, cy_true) = (int(target[0] + target[2] / 2), int(target[1] + target[3] / 2))
				dist = math.sqrt(pow(cx_pred-cx_true, 2)+pow(cy_pred-cy_true, 2))
				if dist > tol:
					FP1 += 1
				else:
					TP += 1
		i += 1
	return (TP, TN, FP1, FP2, FN)

#Return the values of accuracy, precision and recall
def evaluation(y_pred, y_true, tol):
	(TP, TN, FP1, FP2, FN) = outcome(y_pred, y_true, tol)
	try:
		accuracy = (TP + TN) / (TP + TN + FP1 + FP2 + FN)
	except:
		accuracy = 0
	try:
		precision = TP / (TP + FP1 + FP2)
	except:
		precision = 0
	try:
		recall = TP / (TP + FN)
	except:
		recall = 0
	return (accuracy, precision, recall)


try:
	(opts, args) = getopt.getopt(sys.argv[1:], '', [
		'load_weights=',
		'save_weights=',
		'dataDir=',
		'epochs=',
		'tol='
	])
	if len(opts) < 4:
		raise ''
except:
	print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
	print('argument --load_weights is required only if you want to retrain the model')
	exit(1)

paramCount={
	'load_weights': 0,
	'save_weights': 0,
	'dataDir': 0,
	'epochs': 0,
	'tol': 0
}

for (opt, arg) in opts:
	if opt == '--load_weights':
		paramCount['load_weights'] += 1
		load_weights = arg
	elif opt == '--save_weights':
		paramCount['save_weights'] += 1
		save_weights = arg
	elif opt == '--dataDir':
		paramCount['dataDir'] += 1
		dataDir = arg
	elif opt == '--epochs':
		paramCount['epochs'] += 1
		epochs = int(arg)
	elif opt == '--tol':
		paramCount['tol'] += 1
		tol = int(arg)
	else:
		print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
		print('argument --load_weights is required only if you want to retrain the model')
		exit(1)

if paramCount['save_weights'] == 0 or paramCount['dataDir'] == 0 or paramCount['epochs'] == 0 or paramCount['tol'] == 0:
	print('usage: python3 train_TrackNet3.py --load_weights=<previousWeightPath> --save_weights=<newWeightPath> --dataDir=<npyDataDirectory> --epochs=<trainingEpochs> --tol=<toleranceValue>')
	print('argument --load_weights is required only if you want to retrain the model')
	exit(1)

#Loss function
def custom_loss(y_true, y_pred): #hm_true, hm_pred
	#pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
	#neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
	#neg_weights = tf.pow(1 - hm_true, 4)

	#pos_loss = -tf.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, 2) * pos_mask
	#neg_loss = -tf.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, 2) * neg_weights * neg_mask

	#num_pos = tf.reduce_sum(pos_mask)
	#pos_loss = tf.reduce_sum(pos_loss)
	#neg_loss = tf.reduce_sum(neg_loss)

	#loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
	loss = (-1)*(K.square(1 - y_pred) * y_true * K.log(K.clip(y_pred, K.epsilon(), 1)) + K.square(y_pred) * (1 - y_true) * K.log(K.clip(1 - y_pred, K.epsilon(), 1)))

	#gamma=2
	#alpha=0.25
	#y_true = tf.cast(y_true, tf.float32)
	#alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
	#p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
	#focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
	#return K.mean(focal_loss)
	return (loss)

#--------------------Weighted hausdorff-distance------------------------------------------------
# 	resized_height = 288  
# 	resized_width  = 512
# 	max_dist = math.sqrt(resized_height**2 + resized_width**2)
# 	n_pixels = resized_height * resized_width
# 	all_img_locations = tf.convert_to_tensor(cartesian([np.arange(resized_height), np.arange(resized_width)]),
# 													tf.float32)


# 	def tf_repeat(tensor, repeats):
# 		"""
# 		Args:
# 		input: A Tensor. 1-D or higher.
# 		repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
# 		Returns:
		
# 		A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
# 		"""
# 		with tf.variable_scope("repeat"):
# 			expanded_tensor = tf.expand_dims(tensor, -1)
# 			multiples = [1] + repeats
# 			tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
# 			repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
# 		return repeated_tesnor


# 	terms_1 = []
# 	terms_2 = []
# 	y_true = tf.squeeze(y_true, axis=-1)
# 	y_pred = tf.squeeze(y_pred, axis=-1)
# #     y_true = tf.reduce_mean(y_true, axis=-1)
# #     y_pred = tf.reduce_mean(y_pred, axis=-1)
# 	for b in range(batch_size=BATCH_SIZE):
# 		gt_b = y_true[b]
# 		prob_map_b = y_pred[b]
# 		# Pairwise distances between all possible locations and the GTed locations
# 		n_gt_pts = tf.reduce_sum(gt_b)
# 		gt_b = tf.where(tf.cast(gt_b, tf.bool))
# 		gt_b = tf.cast(gt_b, tf.float32)
# 		d_matrix = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(gt_b*gt_b, axis=1), (-1, 1)) + tf.reduce_sum(all_img_locations*all_img_locations, axis=1)-2*(tf.matmul(gt_b, tf.transpose(all_img_locations))), 0.0))
# 		d_matrix = tf.transpose(d_matrix)
# 		# Reshape probability map as a long column vector,
# 		# and prepare it for multiplication
# 		p = tf.reshape(prob_map_b, (n_pixels, 1))
# 		n_est_pts = tf.reduce_sum(p)
# 		p_replicated = tf_repeat(tf.reshape(p, (-1, 1)), [1, n_gt_pts])
# 		eps = 1e-6
# 		alpha = 4
# 		# Weighted Hausdorff Distance
# 		term_1 = (1 / (n_est_pts + eps)) * tf.reduce_sum(p * tf.reshape(tf.reduce_min(d_matrix, axis=1), (-1, 1)))
# 		d_div_p = tf.reduce_min((d_matrix + eps) / (p_replicated**alpha + eps / max_dist), axis=0)
# 		d_div_p = tf.clip_by_value(d_div_p, 0, max_dist)
# 		term_2 = tf.reduce_mean(d_div_p, axis=0)
# 		terms_1.append(term_1)
# 		terms_2.append(term_2)
# 	terms_1 = tf.stack(terms_1)
# 	terms_2 = tf.stack(terms_2)
# 	terms_1 = tf.Print(tf.reduce_mean(terms_1), [tf.reduce_mean(terms_1)], "term 1")
# 	terms_2 = tf.Print(tf.reduce_mean(terms_2), [tf.reduce_mean(terms_2)], "term 2")
# 	res = terms_1 + terms_2
# 	return res
#--------------------Weighted hausdorff-distance------------------------------------------------


#Training for the first time
if paramCount['load_weights'] == 0:
	model=TrackNet3(HEIGHT, WIDTH)
	ADADELTA = optimizers.Adadelta(lr=1.0)
	model.compile(loss=custom_loss, optimizer=ADADELTA, metrics=['accuracy'])
#Retraining
else:
	model = load_model(load_weights, custom_objects={'custom_loss':custom_loss})

r = os.path.abspath(os.path.join(dataDir))
path = glob(os.path.join(r, '*.npy'))
num = len(path) / 2
idx = np.arange(num, dtype='int') + 1
print('Beginning training......')
loss_list=[]
for i in range(epochs):
	print('============epoch', i+1, '================')
	loss = 0 ###############
	np.random.shuffle(idx)
	for j in idx:
		x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
		y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
		history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1)
		#############
		loss += history.history['loss'][0] ##################
		del x_train
		del y_train
	loss_list.append(loss)
	#Show the outcome of training data so long
	TP = TN = FP1 = FP2 = FN = 0
	for j in idx:
		x_train = np.load(os.path.abspath(os.path.join(dataDir, 'x_data_' + str(j) + '.npy')))
		y_train = np.load(os.path.abspath(os.path.join(dataDir, 'y_data_' + str(j) + '.npy')))
		y_pred = model.predict(x_train, batch_size=BATCH_SIZE)
		y_pred = y_pred > 0.5
		y_pred = y_pred.astype('float32')
		(tp, tn, fp1, fp2, fn) = outcome(y_pred, y_train, tol)
		TP += tp
		TN += tn
		FP1 += fp1
		FP2 += fp2
		FN += fn
		del x_train
		del y_train
		del y_pred
	print("Outcome of training data of epoch " + str(i+1) + ":")
	print("Number of true positive:", TP)
	print("Number of true negative:", TN)
	print("Number of false positive FP1:", FP1)
	print("Number of false positive FP2:", FP2)
	print("Number of false negative:", FN)
	#Save intermediate weights during training
	if (i + 1) % 1 == 0:
		model.save(save_weights + '/mymodel_' + str(i + 1))

print('Saving weights......')
model.save(save_weights)

#############################
title = 'Model Loss'
plt.title(title)
plt.xlabel('epoch')
plt.ylabel('loss')
x = np.arange(1,epochs+1,1)
plt.plot(x, loss_list)
plt.savefig(title + '.jpg')
#############################

print('Done......')
