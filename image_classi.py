#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test)=cifar10.load_data()
print('xtrash',x_train.shape)


# In[9]:





# In[5]:


import matplotlib.pyplot as plt
img=plt.imshow(x_train[0])


# In[6]:


y_train[0]


# In[7]:


from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
print(y_train_one_hot)


# In[10]:


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255
x_test = x_test / 255


# In[11]:


x_train.shape


# In[1]:


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout,Activation
from tensorflow.keras import layers


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))


# In[15]:


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[16]:


hist = model.fit(x_train, y_train_one_hot, 
           batch_size=256, epochs=10, validation_split=0.2,shuffle=True )


# In[17]:


model.evaluate(x_test, y_test_one_hot)[1]


# In[1]:


new_image = plt.imread("cat2.jpg")
from skimage.transform import resize
resized_image = resize(new_image, (32,32,3))
img = plt.imshow(resized_image)

import numpy as np
probabilities = model.predict(np.array( [resized_image] ))

index=probabilities

classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

index=np.argsort(probabilities[0:])
print(index)

