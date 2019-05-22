#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:05:48 2018

@author: ylu25
"""

'''
    apply all classes MMAC dataset, include IMU data to form bi-model system, model generator output single model
'''

import numpy as np
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
import os
import argparse
from keras import callbacks
from data_loader import load_data_example
import pdb

K.set_image_data_format('channels_last')


def CapsNet_lstm(imgset_input_shape, n_class, routings):
    img_total_input = layers.Input(shape=imgset_input_shape)    
    y = layers.Input(shape=(n_class,))
    
    for loop_idx in range(imgset_input_shape[0]): 
        x = layers.Lambda(lambda img_total_input: img_total_input[:,loop_idx,:,:,:])(img_total_input)   
        conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu')(x)
        primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
        digitcaps_mid = CapsuleLayer(num_capsule=2 * n_class, dim_capsule=16, routings=routings, squash_flag=False)(primarycaps)
        digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, squash_flag=False)(digitcaps_mid)
        out_caps = Length()(digitcaps)
        fe_ve = layers.Dense(128, activation='relu')(out_caps)

        fe_ve_reshape = layers.Reshape((1,128))(fe_ve)
        if loop_idx == 0:
            total_fe = fe_ve_reshape
        else:
            total_fe = layers.concatenate([total_fe,fe_ve_reshape],axis = 1)

        # Decoder network.
        
        masked_by_y = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer. For training
        masked = Mask()(digitcaps)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = models.Sequential()
        decoder.add(layers.Dense(512, activation='relu', input_dim=16 * n_class))
        decoder.add(layers.Dense(np.prod(imgset_input_shape[1:]), activation='sigmoid'))
        decoder.add(layers.Reshape(target_shape=imgset_input_shape[1:]))
        out_decoder_train = decoder(masked_by_y)
        out_decoder_test = decoder(masked)
        out_decoder_train_reshape = layers.Reshape((1, imgset_input_shape[1], imgset_input_shape[2], imgset_input_shape[3]))(out_decoder_train)
        
        if loop_idx == 0:
            total_decoder_out = out_decoder_train_reshape
        else:
            total_decoder_out = layers.concatenate([total_decoder_out, out_decoder_train_reshape], axis=1)
    lstm = layers.LSTM(256 , input_dim = total_fe.shape[2], input_length=total_fe.shape[1], return_sequences=True, activation='relu')(total_fe)
    lstm = layers.LSTM(64, activation='relu')(lstm)
    fc_img_out = layers.Dense(n_class, activation='softmax')(lstm)
    
    train_model = models.Model([img_total_input, y], [fc_img_out, total_decoder_out])

    return train_model      

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `(x_img_train, x_IMU_train, y_train), (x_img_test, x_IMU_test, y_test, y_test_oncol)`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train, x_test, y_test) = data
    y_train_onehot = to_categorical(y_train, 119)
    y_test_onehot = to_categorical(y_test, 119)

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weight-{epoch:02d}.h5', monitor='val_fc_out_acc',
                                           save_best_only=True, save_weights_only=True , verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'fc_out': 'accuracy'})


    # Training without data augmentation:
    model.fit([x_train, y_train_onehot], [y_train_onehot, x_train], batch_size=args.batch_size, epochs=args.epochs, validation_data=[[x_test, y_test_onehot], [y_test_onehot, x_test]], callbacks=[log, tb,checkpoint, lr_decay], verbose=2)

    return model


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./test')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    (x_train, y_train) = load_data_example()

    # define model: testing data loaded only
    model = CapsNet_lstm(imgset_input_shape = x_train.shape[1:],
                                                  n_class=119,
                                                  routings=3)

    model.summary()

    train(model=model, data=((x_train, y_train, x_train, y_train)), args=args)
