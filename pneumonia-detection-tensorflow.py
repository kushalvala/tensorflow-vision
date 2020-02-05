# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
ls

# %% [code]
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os

# %% [code]
# Reading the directory - Train and Validation Set

train_pneumonia_data = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')
train_normal_data = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL')

test_pneumonia_data = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA')
test_normal_data = os.path.join('/kaggle/input/chest-xray-pneumonia/chest_xray/test/NORMAL')

# %% [code]
# Lets look at the images in train dataset

train_pneumonia_names = os.listdir(train_pneumonia_data)

print(train_pneumonia_names[:10])

train_normal_names = os.listdir(train_normal_data)

print(train_normal_names[:10])

# %% [code]
print('Total training Pneumonia detected images:', len(os.listdir(train_pneumonia_data)))
print('Total training Normal detected images:', len(os.listdir(train_normal_data)))

# %% [code]
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_pn_pix = [os.path.join(train_pneumonia_data, fname) 
                for fname in train_pneumonia_names[pic_index-8:pic_index]]
next_normal_pix = [os.path.join(train_normal_data, fname) 
                for fname in train_normal_names[pic_index-8:pic_index]]

for i, img_path in enumerate(next_pn_pix+next_normal_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

# %% [code]
import cv2
import glob
all_h = []
all_w = []
for img in glob.glob("/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/*"):
    n= cv2.imread(img)
    h, w, _ = n.shape
    all_h.append(h)
    all_w.append(w)

# %% [code]
print('Average Height of the Train Data:', np.average(all_h))
print('Average Width of the Train data:' , np.average(all_w))

# %% [code]
# Model Building:

model = tf.keras.models.Sequential( [
    # First Convolution
    tf.keras.layers.Conv2D(16 , (3,3) , activation = 'relu' , input_shape = (825,1200,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # Second Convolution
    tf.keras.layers.Conv2D(4, (3,3) , activation = 'relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(0.2),
    # Flatten the images
    tf.keras.layers.Flatten(),
    #tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(0.2),
    tf.keras.layers.Dense(1 , activation = 'sigmoid')
]    
)

# %% [code]
model.summary()

# %% [code]
from tensorflow.keras.optimizers import RMSprop

model.compile(loss= 'binary_crossentropy' , optimizer= RMSprop(lr = 0.05) , metrics= ['acc'])

# %% [code]
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1/255)

# %% [code]
train_generator = train_datagen.flow_from_directory(
        '/kaggle/input/chest-xray-pneumonia/chest_xray/train/',  # This is the source directory for training images
        target_size=(825, 1200),  # All images will be resized to 150x150
        #batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# Flow training images in batches of 128 using train_datagen generator
test_generator = test_datagen.flow_from_directory(
        '/kaggle/input/chest-xray-pneumonia/chest_xray/test/',  # This is the source directory for training images
        target_size=(825, 1200),  # All images will be resized to 150x150
        #batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

# %% [code]
history = model.fit_generator(
      train_generator,
      steps_per_epoch=8,  
      epochs=20,
      verbose=1,
      validation_data = test_generator,
      validation_steps=2)

# %% [code]
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss'] 
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# %% [code]
