
# coding: utf-8

# In[1]:

import random
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os
import random
import math

from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Dense,GlobalAveragePooling2D,MaxPooling2D,Flatten
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, Dropout
from keras.layers import Lambda, Input
from sklearn.metrics import accuracy_score
from keras.layers import BatchNormalization
from keras.optimizers import rmsprop
from keras.optimizers import Adadelta,Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


# In[2]:

def euclidean_distance(vects):
    '''
    Auxiliary function to compute the Euclidian distance between two vectors
    in a Keras layer.
    '''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
    #return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


# In[3]:

def contrastive_loss(y_true, y_pred):
    '''
    Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    @param
        y_true : true label 1 for positive pair, 0 for negative pair
        y_pred : distance output of the Siamese network
    '''
    margin = 1
    # if positive pair, y_true is 1, penalize for large distance returned by Siamese network
    # if negative pair, y_true is 0, penalize for distance smaller than the margin
    return K.mean(y_true * K.square(y_pred) +
                  (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


# In[4]:

def data_load():
    paired_pics = []
    unpaired_pics = []
    y_labels = []
    size = (224, 224, 3)
    
    with open("finaltrain_pairs.txt", "r") as f:
        paired_list = [eval(i.strip()) for i in f.readlines()]
        
    random.seed(100)
    random.shuffle(paired_list)
    
    with open("finaltrain_unpaired.txt", "r") as f:
        unpaired_list = [eval(i.strip()) for i in f.readlines()]
    
    random.seed(200)
    random.shuffle(unpaired_list)
    
    for i in range(len(paired_list)):
    # paried photos:
        route1 = "./N_Transformed photos/"
        route2 = "./N_Transformed products/"
        # importing
        image_x1 = image.load_img((route1+paired_list[i][0]+".png"), target_size = size)
        image_x2 = image.load_img((route2+str(paired_list[i][1])+".png"), target_size = size)
        # converting to array
        image1 = image.img_to_array(image_x1)
        image2 = image.img_to_array(image_x2)
        x1 = preprocess_input(image1)
        x2 = preprocess_input(image2)
        paired_pics.append([x1, x2])

    # unpaired photos:
        # importing
        image_x3 = image.load_img((route1+unpaired_list[i][0]+".png"), target_size = size)
        image_x4 = image.load_img((route2+str(unpaired_list[i][1])+".png"), target_size = size)
        # converting to array
        image3 = image.img_to_array(image_x3)
        image4 = image.img_to_array(image_x4)
        x3 = preprocess_input(image3)
        x4 = preprocess_input(image4)
        paired_pics.append([x3, x4])
        
        y_labels += [1, 0]
        
    return np.array(paired_pics), np.array(y_labels).reshape(-1, 1)


# In[5]:

def compute_accuracy(predictions, labels):
    '''
    Compute classification accuracy with a fixed threshold on distances.
    @param 
        predictions : values computed by the Siamese network
        labels : 1 for positive pair, 0 otherwise
    '''
    # the formula below, compute only the true positive rate]
    #    return labels[predictions.ravel() < 0.5].mean()
    n = labels.shape[0]
    acc = 0
    for i in range(len(labels)):
        if (labels[i] == 1 and predictions[i] < 0.5) or (labels[i]==0 and predictions[i]>=0.5):
            acc += 1
    acc = float(acc)/float(n)
    """
    acc =  float(labels[predictions.ravel() < 0.5].sum() +  # count True Positive
               (1-labels[predictions.ravel() >= 0.5]).sum() ) / float(n)  # True Negative
    """

    return acc


# In[6]:

def create_model(main_input_a,main_input_b):
    
    model_vgg_a = VGG16(include_top=False,weights = 'imagenet',input_tensor=main_input_a)
    model_vgg_a.name = 'vgg_a'
    for layer in model_vgg_a.layers:
        layer.name='vgg_a_'+layer.name 
    
    model_vgg_b = VGG16(include_top=False,weights = 'imagenet',input_tensor=main_input_b)
    model_vgg_b.name = 'vgg_b'
    for layer in model_vgg_b.layers:
        layer.name='vgg_b_'+layer.name 
    
    model_vgg_a.trainable = True
    model_vgg_b.trainable = True
    
    x_a = GlobalAveragePooling2D()(model_vgg_a.output)
    x_a = BatchNormalization()(x_a)
    x_a = Dense(2048, activation='relu')(x_a)
    #x_a = Dense(1, activation='sigmoid')(x_a)
    
    x_b = GlobalAveragePooling2D()(model_vgg_b.output)
    x_b = BatchNormalization()(x_b)
    x_b = Dense(2048, activation='relu')(x_b)
    #x_b = Dense(1, activation='sigmoid')(x_b)
    
    distance = Lambda(euclidean_distance)([x_a,x_b])
    
    distance = Dense(1,activation='sigmoid')(distance)
    
    model = Model(inputs=[main_input_a,main_input_b],outputs=[distance])
    
    return(model)


# In[7]:

def simplistic_solution(head, tail, batch_size=16, epochs=30, model_path=None, training = True):
 
    def step_decay(epoch):
        initial_lrate=0.0001
        drop = 0.1
        epochs_drop=20
        lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate
    
    x_train, y_train = data_load()
    
    input_shape = (224, 224, 3)

    model = create_model(Input(input_shape), Input(input_shape))

    #print (te_pairs[:, 0].shape)
    #print (te_pairs[:, 1].shape)
    #train
    
    if model_path:
        model.load_weights(model_path)
    
    if training:
        ada = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        model.compile(loss=contrastive_loss,
                      metrics=['accuracy'],
                      optimizer=ada)
        lrate = LearningRateScheduler(step_decay)
        checkpointer = ModelCheckpoint('1210-23-adam-weights.{epoch:02d}-{val_loss:.2f}.hdf5',verbose=1,save_best_only=True)
        history = model.fit([x_train[head:tail, 0], x_train[head:tail, 1]], y_train[head:tail],
                  batch_size=batch_size,
                  epochs=epochs,validation_data=([x_train[12000:,0],x_train[12000:,1]],y_train[12000:]),
                  callbacks=[lrate,checkpointer])
        
        print(history.history)

        # compute final accuracy on training and test sets
        pred = model.predict([x_train[head:tail, 0], x_train[head:tail, 1]])
        tr_acc = compute_accuracy(pred, y_train[head:tail])
        pred = model.predict([x_train[12000:, 0], x_train[12000:, 1]])
        te_acc = compute_accuracy(pred, y_train[12000:])

        print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    return (model)


# In[ ]:

model_results = simplistic_solution(6000,12000,batch_size=30,epochs=30,model_path='1210-22-adam-weights.21-0.19.hdf5')

"""
x_train, y_train = data_load()
pred = model_results.predict([x_train[:6000, 0], x_train[:6000, 1]])
tr_acc = compute_accuracy(pred, y_train[:6000])
pred = model_results.predict([x_train[12000:, 0], x_train[12000:, 1]])
te_acc = compute_accuracy(pred, y_train[12000:])
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
"""


# In[ ]:



