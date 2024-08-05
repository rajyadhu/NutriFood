#!/usr/bin/env python
# coding: utf-8

# In[2]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# In[4]:


data_train_path=r"D:\Datasets\kaggle\training"


# In[5]:


data_val_path=r'D:\Datasets\kaggle\validation'


# In[6]:


data_test_path=r'D:\Datasets\kaggle\evaluation'


# In[7]:


img_width = 180
img_height = 180


# In[8]:


data_train = tf.keras.utils.image_dataset_from_directory(
    data_train_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)


# In[9]:


data_train.class_names


# In[10]:


data_category = data_train.class_names


# In[11]:


data_val = tf.keras.utils.image_dataset_from_directory(
    data_val_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)


# In[12]:


data_test = tf.keras.utils.image_dataset_from_directory(
    data_test_path,
    shuffle=True,
    image_size=(img_width,img_height),
    batch_size=32,
    validation_split=False)


# In[13]:


plt.figure(figsize=(10,10))
for image,labels in data_train.take(1):
    for i in range(9):
        plt.subplot(5,5,i+1)
        plt.imshow(image[i].numpy().astype('uint8'))
        plt.title(data_category[labels[i]])
        plt.axis('off')


# In[14]:


from tensorflow.keras.models import Sequential


# In[15]:


data_train


# In[16]:


model = Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3, padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(128),
    layers.Dense(len(data_category))
])


# In[18]:


model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[ ]:


epochs_size = 25
history = model.fit(data_train, validation_data=data_val, epochs=epochs_size)


# In[ ]:


epochs_range = range(epochs_size)
plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range,history.history['accuracy'],label = 'Training Accuracy')
plt.plot(epochs_range, history.history['val_accuracy'],label = 'Validation Accuracy')
plt.title('Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,history.history['loss'],label = 'Training Loss')
plt.plot(epochs_range, history.history['val_loss'],label = 'Validation Loss')
plt.title('Loss')


# In[ ]:


image = r'C:\Users\yadhu\Downloads\macaroni-noodles-with-meat-tomato-sauce-served-plate-table_1220-6904.avif'
#print(image)
#plt.imshow(image.astype('uint8'))
image = tf.keras.utils.load_img(image, target_size=(img_height,img_width))
img_arr = tf.keras.utils.array_to_img(image)
img_bat=tf.expand_dims(img_arr,0)


# In[ ]:


predict = model.predict(img_bat)


# In[ ]:


score = tf.nn.softmax(predict)


# In[ ]:


print('Veg/Fruit in image is {} with accuracy of {:0.2f}'.format(data_category[np.argmax(score)],np.max(score)*100))


# In[ ]:





# In[ ]:




