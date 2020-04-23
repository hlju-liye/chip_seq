#coding:utf8
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten, Activation
from keras.optimizers import Adam

model = Sequential()

model.add(Convolution2D(filters = 32, kernel_size = (4, 10), strides = 1,
					padding = 'same', input_shape = (1, 4, 300),
					data_format = 'channels_first'))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (1, 2), strides = 2, padding = 'same',
					data_format = 'channels_first'))

model.add(Convolution2D(filters = 64, kernel_size = (4, 10), strides = 1,
					padding = 'same'))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (1, 2), strides = 2, padding = 'same',
					data_format = 'channels_first'))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(optimizer = Adam(lr = 1e-4),
			loss = 'categorical_crossentropy',
			metrics = ['accuracy'])

