import numpy as np
import pandas as pd
import math
import os
import pathlib
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Layer, ReLU, LeakyReLU
from tensorflow.keras import backend as K
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler,RobustScaler,PowerTransformer,MinMaxScaler,Normalizer



import matplotlib.pyplot as plt
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10,8),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)



filename = "background.csv"
df = pd.read_csv(filename)
df = df.drop(columns=['Unnamed: 0'],axis=1)

#scaler = Normalizer()

for c in df.columns:
    df[c] = (df[c] - df[c].mean())/df[c].std()


df = df.sample(frac = 1)
sample_frac = .6

X_train = df[:int(sample_frac*df.shape[0])].values
X_val = df[int(sample_frac*df.shape[0]):int((sample_frac+.2)*df.shape[0])].values
X_test = df[int((sample_frac+.2)*df.shape[0]):].values


#scaler.fit_transform(X_train)
#scaler.transform(X_val)
#scaler.transform(X_test)

def Model():
    nodes = [32,24,16]
    latent_space_dim = 5
    activation="LeakyReLU"
    model = keras.Sequential([
        #keras.Input(shape=(57,)),
        keras.layers.Dense(nodes[0],use_bias=False,activation=activation,name='Dense_11'),
        keras.layers.Dense(nodes[1],use_bias=False,activation=activation,name='Dense_12'),
        keras.layers.Dense(nodes[2],use_bias=False,activation="ReLU",name='Dense_13'),
        keras.layers.Dense(latent_space_dim,use_bias=False,activation=activation,name='LatentSpace'),
        keras.layers.Dense(nodes[2],use_bias=False,activation="ReLU",name='Dense_23'),
        keras.layers.Dense(nodes[1],use_bias=False,activation=activation,name='Dense_22'),
        keras.layers.Dense(nodes[0],use_bias=False,activation=activation,name='Dense_21'),
        keras.layers.Dense(X_train.shape[1],use_bias=False),
    ])
    model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.08,beta_1=0.42,beta_2=0.32,epsilon=1e-03,amsgrad=True),metrics=['accuracy'], loss='mse')
    input_shape = X_train.shape
    model.build(input_shape)
    model.summary()
    return model


EPOCHS = 250
BATCH_SIZE = 512
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
    min_delta=.001,
    patience=8,
    verbose=1,
    mode="min",
    baseline=None,
    restore_best_weights=True,
)

history = autoencoder.fit(X_train, X_train, epochs = EPOCHS, batch_size = BATCH_SIZE,
                  validation_data=(X_val, X_val))#,callbacks=callbacks)


bkg_prediction = autoencoder.predict(X_test)

#signal labels
signal_labels = ["signal"]
#signal paths
signals_file = ["signal.csv"]

signal_data = []
for i, label in enumerate(signal_labels):
    df = pd.read_csv(signals_file[i])
    for c in df.columns:
        df[c] = (df[c] - df[c].mean())/df[c].std()
    df = df.drop(columns=['Unnamed: 0'],axis=1)
    test_data = df[:].values
    #scaler.transform(test_data)
    signal_data.append(test_data)


signal_results = []

for i, label in enumerate(signal_labels):
    signal_prediction = autoencoder.predict(signal_data[i])
    signal_results.append(
        [label, signal_data[i], signal_prediction]
    )


def mse_loss(true, prediction):
    loss = tf.reduce_mean(tf.math.square(true - prediction), axis=-1)
    return loss

# compute loss value (true, predicted)
total_loss = []
total_loss.append(mse_loss(X_test, bkg_prediction.astype(np.float32)).numpy())
for i, signal_X in enumerate(signal_data):
    total_loss.append(
        mse_loss(signal_X, signal_results[i][2].astype(np.float32)).numpy()
    )

bin_size = 100

plt.rcParams.update(params)

#plt.figure(figsize=(10, 8))
plt.hist(
        total_loss[0],
        bins=bin_size,
        label="Background",
        density=True,
        histtype="step",
        fill=False,
        linewidth=1.5,
    )
for i, label in enumerate(signal_labels):
    plt.hist(
        total_loss[i+1],
        bins=bin_size,
        label=label,
        density=True,
        histtype="step",
        fill=False,
        linewidth=1.5,
    )
plt.yscale("log")
plt.xlabel("Autoencoder Loss")
plt.ylabel("Probability")
#plt.title("MSE loss")
plt.legend(loc="best")
plt.savefig('dnn/mse_loss_dnn.pdf')
plt.show()


labels = np.concatenate([["Background"], np.array(signal_labels)])

epochs = range(1,len(history.history['loss'])+1)
#epochs =range(1,int(len(history.history['loss']+1)))# range(1,Model().params.get('steps')+1)
#

plt.rcParams.update(params)
plt.plot(epochs,history.history["loss"], label="Training loss")
plt.plot(epochs,history.history["val_loss"], label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig('dnn/loss_dnn.pdf')
plt.show()
plt.plot(epochs,history.history["accuracy"], label="Training accuracy")
plt.plot(epochs,history.history["val_accuracy"], label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('dnn/accuracy_dnn.pdf')
plt.show()


target_background = np.zeros(total_loss[0].shape[0])

plt.rcParams.update(params)
epsilon = 1e-5
for i, label in enumerate(labels):
    if i == 0:
        continue  # background events
    trueVal = np.concatenate((np.ones(total_loss[i].shape[0]), target_background))
    predVal_loss = np.concatenate((total_loss[i], total_loss[0]))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)
    auc_loss = auc(fpr_loss, tpr_loss)
    df = pd.DataFrame(columns=['FPR','TPR'])
    df['FPR'] = fpr_loss
    df['TPR'] = tpr_loss
    df.to_csv('dnn/roc_'+label+".csv")
    plt.plot(
        tpr_loss,
        1/(fpr_loss+epsilon),
        "-",
        label=f"{label} (AUC = {auc_loss * 100.0:.1f}%)",
        linewidth=1.5,
    )
    plt.yscale('log')
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection (1/$\epsilon_b$)")
    plt.legend(loc="best")
    plt.xlim(1e-2,1)
plt.plot(np.linspace(0, 1), 1/(np.linspace(0, 1)+epsilon), "--", color="0.75")
plt.savefig('dnn/roc_dnn.pdf')
plt.show()



for i, label in enumerate(labels):
    if i == 0:
        continue  # background events
    trueVal = np.concatenate((np.ones(total_loss[i].shape[0]), target_background))
    predVal_loss = np.concatenate((total_loss[i], total_loss[0]))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(trueVal, predVal_loss)
    auc_loss = auc(fpr_loss, tpr_loss)
    plt.plot(
        tpr_loss,
        1-fpr_loss,
        "-",
        label=f"{label} (AUC = {auc_loss * 100.0:.1f}%)",
        linewidth=1.5,
    )
    #plt.yscale('log')
    plt.xlabel("Signal Efficiency")
    plt.ylabel(f"Background Rejection (1-$\epsilon_b$)")
    plt.legend(loc="best")
    plt.xlim(1e-2,1)
plt.plot(np.linspace(0, 1), 1-(np.linspace(0, 1)), "--", color="0.75")
plt.savefig('dnn/roc_dnn_2.pdf')
plt.show()
