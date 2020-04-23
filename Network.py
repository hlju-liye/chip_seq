#coding:utf-8

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Activation, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical

from scipy.ndimage import convolve

import numpy as np

def conv2(m1, m2):
	sum = 0
	for i in np.arange(4):
		for j in np.arange(15):
			sum = sum + m1[i][j] * m2[i][j]
	return sum

def transform_matrix_to_seq(matrix_array):
	indexp = 1
	for seqs_for_one_filter in matrix_array:
		motif = open('motif_sequence_' + str(indexp) + '.txt', 'wr')
		indexq = 1
		for seq in seqs_for_one_filter:
			motif.write('>' + str(indexq))
			motif.write('\r\n')
			for i in np.arange(15):
				for j in np.arange(4):
					if seq[j][i] == 1:
						if j == 0:
							motif.write('A')
						elif j == 1:
							motif.write('C')
						elif j == 2:
							motif.write('G')
						elif j == 3:
							motif.write('T')
			motif.write('\r\n')
			indexq = indexq + 1
		indexp = indexp + 1


class Network:

	model = []

	def __init__(self):
		self.model = Sequential()

		self.model.add(Conv2D(filters = 10, kernel_size = (4, 15), strides = 1,
						padding = 'same', input_shape = (4, 300, 1),
						data_format = 'channels_last'))

		self.model.add(Activation('relu'))

		self.model.add(MaxPooling2D(pool_size = (1, 5), strides = 5, padding = 'same',
					))

		self.model.add(Conv2D(filters = 128, kernel_size = (1, 15), strides = 1,
						padding = 'same'))

		self.model.add(Activation('relu'))

		self.model.add(MaxPooling2D(pool_size = (1,10), strides = 10, padding = 'same',
					))

		self.model.add(Flatten())

		self.model.add(Dense(1024))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(128))
		self.model.add(Activation('relu'))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(1))
		self.model.add(Activation('sigmoid'))

		self.model.compile(optimizer = Adam(lr = 1e-4),
					loss = 'binary_crossentropy',
					metrics = ['accuracy'])

	def train(self, x_train, y_train, split):
		self.model.fit(x_train, y_train, 
			validation_split = split, epochs = 3)

	def evaluate(self, x_test, y_test):

		#print len(self.model.layers[0].get_weights())
		
		#print self.model.layers[0].get_weights()[0][:,:,:,0].reshape(4, 15)
		print x_test.shape
		all_filter_weights = self.model.layers[0].get_weights()[0].squeeze()
		print all_filter_weights.shape
		filter_count = all_filter_weights.shape[2]
		print filter_count
		EAV = np.zeros(filter_count)

		test_seqs = x_test.squeeze()

		test_seqs_count = x_test.shape[0]
		print test_seqs_count
		scan_count = x_test.shape[2] - all_filter_weights.shape[1]
		print scan_count

		#先找出每个filter的EAV值
		for filter_index in np.arange(filter_count):

			filter_weights = all_filter_weights[:,:,filter_index]

			for seqs_count_index in np.arange(test_seqs_count):

				test_seq = x_test[seqs_count_index].squeeze()

				for scan_index in np.arange(scan_count):

					conv_value = conv2(filter_weights, test_seq[:, scan_index:scan_index + 15])
					if conv_value > EAV[filter_index]:
						EAV[filter_index] = conv_value

			print EAV[filter_index]

		#找出指定阈值的sequence
		candidate_seqs = []
		for filter_index in np.arange(filter_count):

			filter_weights = all_filter_weights[:,:,filter_index]
			seqs = []
			for seqs_count_index in np.arange(test_seqs_count):

				test_seq = x_test[seqs_count_index].squeeze()

				for scan_index in np.arange(scan_count):

					conv_value = conv2(filter_weights, test_seq[:, scan_index:scan_index + 15])
					if conv_value > EAV[filter_index] * 0.9:
						seqs.append(test_seq[:, scan_index:scan_index + 15])

			candidate_seqs.append(seqs)


		print(len(candidate_seqs))
		transform_matrix_to_seq(candidate_seqs)
		print EAV
		return self.model.evaluate(x_test, y_test)

	def predict(self, x_test):
		return self.model.predict(x_test)


		

	
