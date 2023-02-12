import numpy as np
import pandas as pd
import math
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Layer, ReLU, LeakyReLU
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


filename = "background.csv"
df = pd.read_csv(filename)

df = df.sample(frac = 1)

sample_frac = .6

X_train = df[:int(sample_frac*df.shape[0])].values
X_val = df[int(sample_frac*df.shape[0]):int((sample_frac+.2)*df.shape[0])].values
X_test = df[int((sample_frac+.2)*df.shape[0]):].values


def Model():
    nodes = [18,15,12]
    latent_space_dim = 4
    activation="LeakyReLU"
    model = keras.Sequential([
        #keras.Input(shape=(57,)),
        keras.layers.Dense(nodes[0],use_bias=False,activation=activation,name='Dense_11'),
        keras.layers.Dense(nodes[1],use_bias=False,activation=activation,name='Dense_12'),
        keras.layers.Dense(nodes[2],use_bias=False,activation=activation,name='Dense_13'),
        keras.layers.Dense(latent_space_dim,use_bias=False,activation=activation,name='LatentSpace'),
        keras.layers.Dense(nodes[2],use_bias=False,activation=activation,name='Dense_23'),
        keras.layers.Dense(nodes[1],use_bias=False,activation=activation,name='Dense_22'),
        keras.layers.Dense(nodes[0],use_bias=False,activation=activation,name='Dense_21'),
        keras.layers.Dense(X_train.shape[1],use_bias=False),
    ])
    model.compile(optimizer = keras.optimizers.Adam(),metrics=['accuracy'], loss='mse')
    input_shape = X_train.shape
    model.build(input_shape)
    model.summary()
    return model


EPOCHS = 30
BATCH_SIZE = 256
autoencoder = Model()#inputs = inputArray, outputs=decoder

'''
tf.keras.utils.plot_model(
    autoencoder,
    to_file='result/model_arch.png',
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir='TB',
    expand_nested=True,
    dpi=96,
    layer_range=None,
    show_layer_activations=False
)
'''

callbacks = tf.keras.callbacks.EarlyStopping(
    monitor="loss",
    min_delta=0.002,
    patience=5,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)
history = autoencoder.fit(X_train, X_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
                  validation_data=(X_val, X_val))


bkg_prediction = autoencoder.predict(X_test)
