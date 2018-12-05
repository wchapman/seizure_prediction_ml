"""
Runs the finalized ARIMA-coefficient based model on all subjects. This uses a fifth-order multivariate AR model to
fit each trial (sampled at 40Hz) seperately. Coefficients are then passed into a neural network for classification.
"""

# %% Imports
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split
import keras
import keras.layers as layers
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import svm

sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml/arima')
import utils
import utils

%matplotlib auto
# %% Load coefficients
plt.figure()
models = list()
for pat in range(1, 4):
    coefs = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_coefs2',"rb"))
    y = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_y',"rb"))
    X = coefs[-1]

    if X.shape[1] > 1:
        X_flat = np.zeros((X.shape[0],                             # n_trials
                           X.shape[1],                             # n_blocks
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets
        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial][block] = X[trial][block].flatten()
    else:
        X_flat = np.zeros((X.shape[0],
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets

        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial] = X[trial].flatten()

    # %%

    model = keras.Sequential()
    model.add(layers.Dense(32,
                          input_dim=1280))

    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(64))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(32))
    model.add(layers.Dropout(0.2))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    x_train, x_test, y_train, y_test = train_test_split(X_flat, y, stratify=y)

    model.fit(x_train, y_train, batch_size=32, epochs=10000)

    models.append(model)

    # %% Get ROC

    logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
    print(logit_roc_auc)
    logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' %(pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    #plt.savefig('final_ROC')

# %% Logistic Regression

plt.figure()
for pat in range(1, 4):
    coefs = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_coefs2',"rb"))
    y = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_y',"rb"))
    X = coefs[-1]

    if X.shape[1] > 1:
        X_flat = np.zeros((X.shape[0],                             # n_trials
                           X.shape[1],                             # n_blocks
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets
        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial][block] = X[trial][block].flatten()
    else:
        X_flat = np.zeros((X.shape[0],
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets

        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial] = X[trial].flatten()


    x_train, x_test, y_train, y_test = train_test_split(X_flat, y, stratify=y)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    logit_roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])
    print(logit_roc_auc)

    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:,1])
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' % (pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

# %% SVM

plt.figure()
for pat in range(1, 4):
    coefs = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_coefs2',"rb"))
    y = pickle.load(open('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_y',"rb"))
    X = coefs[-1]

    if X.shape[1] > 1:
        X_flat = np.zeros((X.shape[0],                             # n_trials
                           X.shape[1],                             # n_blocks
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets
        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial][block] = X[trial][block].flatten()
    else:
        X_flat = np.zeros((X.shape[0],
                           X.shape[2] * X.shape[3] * X.shape[4]))  # n_fets

        for trial in range(0, X_flat.shape[0]):
            for block in range(0, X_flat.shape[1]):
                X_flat[trial] = X[trial].flatten()


    x_train, x_test, y_train, y_test = train_test_split(X_flat, y, stratify=y)
    clf = svm.SVC(kernel='linear',probability=True)
    clf.fit(x_train, y_train)
    logit_roc_auc = roc_auc_score(y_test, clf.predict_proba(x_test)[:,1])
    print(logit_roc_auc)

    fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(x_test)[:,1])
    plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' % (pat, logit_roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
