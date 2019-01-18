import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Flatten,Dense,Dropout,Activation,Input,Conv2D,Maxpooling2D
from keras.losses import categorical_crossentropy


def alexnet():
	model = Sequential()
	#convolution layer 1
	model.add(Conv2D(96,kernel_size=(11,11),strides=(4,4),padding='valid', data_format = 'channels_last', input_shape = (227, 227, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 2
	model.add(Conv2D(filters=256,kernel_size=(11,11),strides=(1,1),padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	#Convolution layer 3
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#Convolution layer 4
	model.add(Conv2D(filters=384,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(BatchNormalization())

	#convolution layer 5
	model.add(Conv2D(filters=256,kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
	model.add(BatchNormalization())

	model.add(Flatten())

	#Fully connected layer 1
	model.add(Dense(4096, input_shape = (227, 227, 3), activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully  connected layer 2
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.4))
	model.add(BatchNormalization())

	#Fully connected layer 3
	model.add(Dense(38,activation='relu')) # Output size of 38 classes
	model.add(Dropout(0.4))

	#output layer
	model.add(Activation('softmax'))

	model.compile(loss = 'categorical_crossentropy' ,optimizer='adam', metrics=['accuracy']) 

	return model 


def Vgg16():
	model = Sequential()

	#Convolution layer 1
	model.add(Conv2D(64,(3,3), padding='same',))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Maxpooling2D(pool_size=(2,2),strides=(2,2)))	

	#convolution layer 2
	model.add(Conv2D(128,(3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(128,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Maxpooling2D(pool_size=(2,2),strides=(2,2)))

	#convolution layer 3
	model.add(Conv2D(256,(3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(256, (3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(Maxpooling2D(pool_size=(2,2),strides=(2,2)))

	#convolution layer 4
	model.add(Conv2D(512,(3,3), padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(512,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(512,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Maxpooling2D(pool_size=(2,2),strides=(2,2)))

	#convolution layer 5
	model.add(Conv2D(512,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(512,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Conv2D(512,(3,3),padding='same'))
	model.add(Activation('relu'))
	model.add(Maxpooling2D(pool_size=(2,2),strides=(2,2)))

	model.add(Flatten())

	#fully connected layer 1
	model.add(Dense(4096))
	model.add(Activation('relu'))

	#fully connected layer 2
	model.add(Dense(4096))
	model.add(Activation('relu'))

	#output
	model.add(Dense(38))
	model.add(Activation('softmax'))

	model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

	return model  


