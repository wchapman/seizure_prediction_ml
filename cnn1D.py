import os
import h5py
import pandas as pd
import numpy as np
import skimage.measure
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.metrics
import keras
import keras.layers as layers

patients = []
labels = []
i=0
skip = 3
files = os.listdir("resized/")
f = []

for filename in files:
    if not filename.startswith("Pat3Train"): #and not filename.startswith("Pat2Train"):
        continue
    #if(i > 10):
    #    break
    if filename[-5] == '0' and skip > 0:
        skip -= 1
        continue
    elif skip < 1 :
        skip = 3
        
    i+=1
    f.append(filename)

for filename in f:
    data = np.load("resized/"+filename)
    patients.append(data)
    labels.append(filename[-5] == '1')

label_array = np.reshape(np.array(labels, dtype=bool),(-1,1))
pat_arr = np.array(patients)
idx = np.array(range(0,pat_arr.shape[0]))
np.random.shuffle(idx)
pat_arr = pat_arr[idx]
label_array = label_array[idx]

labels_filter = label_array.ravel().copy()
# print(labels_filter[0:100])

# print(pat_arr.shape, label_array.shape)

l = 1 * label_array

X_train, X_val, y_train, y_val = train_test_split(pat_arr, l, test_size=0.3)

print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)

nb_features = X_train[0].shape[1]
nb_out = 1
sequence_length = X_train[0].shape[0]

model = keras.Sequential()

# model.add(Reshape((sequence_length, nb_features), input_shape=(input_shape,)))
model.add(layers.Conv1D(
            filters = 100,
            kernel_size = 10,
            activation ='relu',
            input_shape = (sequence_length, nb_features)))
model.add(layers.MaxPooling1D(pool_size = 5))
# model.add(layers.Dropout(rate = 0.5))
model.add(layers.Conv1D(
            filters = 100, 
            kernel_size = 10, 
            activation='relu'))
model.add(layers.MaxPooling1D(pool_size = 5))
model.add(layers.Dropout(rate = 0.5))
model.add(layers.Conv1D(
            filters = 50,
            kernel_size = 10, 
            activation='relu'))
model.add(layers.MaxPooling1D(pool_size = 5))
# model.add(layers.Dropout(rate = 0.5))
model.add(layers.Conv1D(
            filters = 50, 
            kernel_size = 10, 
            activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(rate = 0.5))
model.add(layers.Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

batch_size = 10
epochs = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

loss_train, acc_train = model.evaluate(X_train, y_train)
pred_train = model.predict(X_train)
auc_train = sklearn.metrics.roc_auc_score(y_train, pred_train)

loss_val, acc_val = model.evaluate(X_val, y_val)
pred_val = model.predict(X_val)
auc_val = sklearn.metrics.roc_auc_score(y_val, pred_val)

print("loss_train: ", loss_train, "\nacc_train: ", acc_train, "\nauc_train: ", auc_train)
print("\nloss_val: ", loss_val, "\nacc_val: ", acc_val, "\nauc_val: ", auc_val)