import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob as glb
import os
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from keras import layers, Input, Model
from keras.models import Sequential
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as pp_i 
from keras.layers import RandomFlip, RandomRotation, Dense, Dropout
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

#filtering only images

images_file = 'C:\\Users\\duttd\\Documents\\VS_code_docs\\Image_Classification_withTensorflow\\oxford-iiit-pet'
image_names = [os.path.basename(file) for file in glb.glob(os.path.join(images_file,'*.jpg'))] 

#creating labels

labels = [' '.join(name.split('_')[:-1:]) for name in image_names]

#encoding labels by a function

def label_encode(label):
    if label == 'Abyssinian' : return 0
    elif label == 'Bengal' : return 1
    elif label == 'Birman' : return 2
    elif label == 'Bombay' : return 3
    elif label == 'British Shorthair' : return 4
    elif label == 'Egyptian Mau' : return 5
    elif label == 'american bulldog' : return 6
    elif label == 'american pit bull terrier' : return 7
    elif label == 'basset hound' : return 8
    elif label == 'beagle' : return 9
    elif label == 'boxer' : return 10
    elif label == 'chihuahua' : return 11
    elif label == 'english cocker spaniel' : return 12
    elif label == 'english setter' : return 13
    elif label == 'german shorthaired' : return 14
    elif label == 'great pyrenees' : return 15

# creating lists to store processed data

features = []
labels = []
image_size = (224,224)

#iterating processed data

for name in image_names:
    labels = ' '.join(name.split('_')[:-1:])
    label_encoded = label_encode(labels)
    if label_encoded != None :
        image = load_img(os.path.join(images_file, name))
        image = tf.image.resize_with_pad(img_to_array(image, dtype='uint8'), *image_size).numpy().astype('uint8')
        image = np.array(image)
        features.append(image)
        labels.append(label_encoded)
features_array = np.array(features)
labels_array = np.array(labels)

#one hot encoding

labels_one_hot= pd.get_dummies(labels_array)

#splitting training data(train - 55%, val - 25%, test - 20%)

x_train, x_test, y_train, y_test = train_test_split(features_array, labels_one_hot, test_size= 0.2, random_state=20)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=2)

# prepping model

data_augmentation = Sequential([RandomFlip('HORIZONTAL_AND_VERTICAL'), RandomRotation(0.2)])
prediction_layers = Dense(16,activation='softmax')

resnet_model = ResNet50(include_top=False, pooling='avg', weights='imagenet')
resnet_model.trainable= False
preprocess_input = pp_i

#Building Model

inputs = Input(shape=(224,224,3))
x= data_augmentation(inputs)
x= preprocess_input(x)
x= resnet_model(x ,training=False)
x= Dropout(0.2)(x)

outputs = prediction_layers(x)
model = Model(inputs, outputs)

#compiling model

model.compile(optimizer=Adam(), loss=categorical_crossentropy(), metrics=['accuracy'])

#fitting model

model_history = model.fit(x= x_train, y=y_train, validation_data=(x_val, y_val), epochs=30)

#plotting parameters

acc= model_history.history['accuracy']
val_acc= model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

#plotting accuracy of model

epochs_range = range(30)
plt.figure(figsize=(15,8))

plt.subplot(1,2,1)
plt.plot(epochs_range,acc,label='training_accuracy')
plt.plot(epochs_range,val_acc,label='validation_accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation accuracy')

plt.subplot(1,2,1)
plt.plot(epochs_range,loss,label='training_loss')
plt.plot(epochs_range,val_loss,label='validation_loss')
plt.legend(loc='upper right')
plt.title('Training and Validation loss')

#evaluatiing against test data set

model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)
print(y_pred)