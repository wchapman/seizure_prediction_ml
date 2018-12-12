"""
Runs the finalized ARIMA-coefficient based model on all subjects. This uses a fifth-order multivariate AR model to
fit each trial (sampled at 40Hz) seperately. Coefficients are then passed into a neural network for classification.
"""

# %% Imports
import numpy as np
import sys

from sklearn.model_selection import train_test_split
import keras
import keras.layers as layers
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml/features')



plt.figure()

# %% LSTM approach
n_blocks=20
plt.cla()
for pat in range(1, 4):

    fname = '/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/fets_' +str(pat) +'_' + str(n_blocks) +'.npz'
    dat = np.load(fname)
    standard_fets = dat['standard_fets']
    ar_fets = dat['ar_fets']
    y = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s' +str(pat) +'_y',"rb"))
    X = ar_fets

    if X.shape[1] > 1:
        X_flat = np.zeros((X.shape[0],                             # n_trials
                           X.shape[1],                             # n_blocks
                           X.shape[2]))  # n_fets
        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial][block] = X[trial][block].flatten()
    else:
        X_flat = np.zeros((X.shape[0],
                           X.shape[2]))  # n_fets

        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial] = X[trial].flatten()

    x_train, x_test, y_train, y_test = train_test_split(X_flat, y, stratify=y)

    ################# %
    dropout = 0.2
    l1 = 0.
    l2 = 0.
    sequence_length = 20
    nb_features = 1280

    model = keras.Sequential()
    model.add(layers.TimeDistributed(
                layers.Dense(64, input_dim=1280)))

    model.add(layers.Dropout(dropout))

    model.add(layers.TimeDistributed(
            layers.Dense(32)))

    model.add(layers.Dropout(dropout))


    model.add(layers.LSTM(units=32, return_sequences=True,
                          kernel_regularizer=keras.regularizers.l1_l2(l1, l2)))

    model.add(layers.SimpleRNN(units=32, return_sequences=False,
                               kernel_regularizer=keras.regularizers.l1_l2(l1, l2)))

    model.add(layers.Dropout(dropout))

    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=1000)

    #################### %% Get ROC

    logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
    logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
    print(logit_roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' %(pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('NN: 0.87')
    plt.legend(loc="lower right")

# %% 1D convolution approach
plt.cla()
for pat in range(1,4):
    n_blocks=20


    fname = '/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/fets_' +str(pat) +'_' + str(n_blocks) +'.npz'
    dat = np.load(fname)
    standard_fets = dat['standard_fets']
    ar_fets = dat['ar_fets']
    y = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s' +str(pat) +'_y',"rb"))
    X = ar_fets



    # %
    if X.shape[1] > 1:
        X_flat = np.zeros((X.shape[0],                             # n_trials
                           X.shape[1],                             # n_blocks
                           X.shape[2]))  # n_fets
        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial][block] = X[trial][block].flatten()
    else:
        X_flat = np.zeros((X.shape[0],
                           X.shape[2]))  # n_fets

        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial] = X[trial].flatten()

    x_train, x_test, y_train, y_test = train_test_split(X_flat, y, stratify=y)

    # %
    dropout = 0.2
    l1 = 0.
    l2 = 0.01
    sequence_length = 20
    nb_features = 1280

    model = keras.Sequential()

    model.add(layers.Conv1D(32, 10, activation='relu', bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            input_shape=(X.shape[1], X.shape[2])))

    model.add(layers.Conv1D(64, 5, activation='relu',
                            bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            ))

    model.add(layers.GlobalAveragePooling1D())


    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0)

    #################### %% Get ROC

    logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
    print(logit_roc_auc)

    logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
    print(logit_roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' %(pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('NN: 0.87')
    plt.legend(loc="lower right")

plt.show()


# %% Doing AR+CNN job
for pat in range(2, 4):
    dat = np.load('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s' +str(pat) + '_40Hz.npz')
    xa = dat['X']
    y = dat['y']
    x_train, x_test, y_train, y_test = train_test_split(xa, y, stratify=y)

    # %
    dropout = 0.2
    l1 = 0.0001
    l2 = 0.
    sequence_length = 20
    nb_features = 1280

    model = keras.Sequential()

    model.add(layers.Conv1D(10, 100, activation='relu', bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            input_shape=(x_train.shape[1], x_train.shape[2])))

    model.add(layers.Dropout(dropout))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(20, 50, activation='relu',
                            bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            ))

    model.add(layers.Dropout(dropout))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(30, 25, activation='relu',
                            bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            ))

    model.add(layers.Dropout(dropout))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(40, 10, activation='relu',
                            bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            ))

    model.add(layers.Dropout(dropout))
    model.add(layers.MaxPooling1D(pool_size=2))

    model.add(layers.Conv1D(50, 5, activation='relu',
                            bias_regularizer=keras.regularizers.l1_l2(l1, l2),
                            ))

    model.add(layers.Dropout(dropout))
    model.add(layers.GlobalAveragePooling1D())


    model.add(layers.Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=32, epochs=10)

    #################### %% Get ROC

    logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
    print(logit_roc_auc)

    logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
    print(logit_roc_auc)
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' % (pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('NN: 0.87')
    plt.legend(loc="lower right")


# %%
block_size = 40
n_blocks = int(xa.shape[1]/block_size)
ar_fets = np.zeros((xa.shape[0], n_blocks, 1280))

import createFets

#% begin
for i in range(0, xa.shape[0]):              # for each trial
    for k in range(0, n_blocks): # for each block
        try:
            inds = np.arange(k*block_size, (k+1)*block_size)
            sf, af = createFets.block_fets(xa[i,inds,:])
        except: # missing data in this block
            af = np.zeros(ar_fets.shape[2])
        ar_fets[i, k, :] = af
