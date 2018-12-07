import numpy as np
import utils
import pyeeg
import matplotlib.pyplot as plt
import pandas as pd
import pdc_dtf
import math

from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.ensemble import RandomForestClassifier

def channel_features(ts, bands, fs, tau, dE):
    em = pyeeg.embed_seq(ts, tau, dE)
    diff1 = pyeeg.first_order_diff(ts)
    singvd = np.linalg.svd(em, compute_uv = 0)
    singvd /= sum(singvd)

    nbands = len(bands)
    nfeatures = 8+nbands # Number of features for each channel (given by pyEEG)

    features_vec = np.zeros(nfeatures)
    features_vec[range(nbands)], pow_ratio = pyeeg.bin_power(ts, bands, fs)
    features_vec[nbands] = pyeeg.hurst(ts)
    features_vec[nbands+1] = pyeeg.pfd(ts, diff1)
    features_vec[range(nbands+2,nbands+4)] = pyeeg.hjorth(ts, diff1)
    features_vec[nbands+4] = pyeeg.spectral_entropy(ts, bands, fs, pow_ratio)
    features_vec[nbands+5] = pyeeg.svd_entropy(ts, tau, dE, singvd)
    features_vec[nbands+6] = pyeeg.fisher_info(ts, tau, dE, singvd)
    features_vec[nbands+7] = pyeeg.dfa(ts)
    

def extract_features(sub):
    data = np.load('/projectnb/cs542/ryanmars/data_downsampled/s'+str(sub)+'_40Hz.npz')

    bands = [0.5, 4, 7, 12, 30] #Frequency bands for delta, theta, alpha, beta, gamma respectively
    fs = 40 #Sampling Rate in Hz
    tau = 10 #embed_seq lag (integer)
    dE = 10 #embedding dimension (integer)

    nfeatures = 8+len(bands) # Number of features for each channel (given by pyEEG)
    nfiles = data['X'].shape[0]
    nchannels = data['X'].shape[2] 

    features_matrix = np.zeros((nfiles, nfeatures * nchannels))
    n_trainpts = math.ceil(0.8*nfiles) # Train on 80% of data, test on 20%

    for file in range(nfiles):
        for channel in range(nchannels):
            ts = data['X'][file,:,channel]
            features_vec = channel_features(ts, bands, fs, tau, dE)
            features_matrix[file,range(channel*nfeatures, (channel+1)*nfeatures-1)] = features_vec

    train_features_matrix = features_matrix[range(n_trainpts),:]
    test_features_matrix = features_matrix[range(n_trainpts,n_files),:]
    train_labels = data['y'][range(n_trainpts)]
    test_labels = data['y'][range(n_trainpts, n_files)]

    return train_labels, train_features_matrix, test_labels, test_features_matrix

train_features_matrix, train_labels, test_featues_matrix, test_labels = extract_features(1)

model = RandomForestClassifier()
print("Training model.")
#train model
model.fit(features_matrix, labels)
predicted_labels = model.predict(test_feature_matrix)
print("FINISHED classifying. accuracy score : ")
print(accuracy_score(test_labels, predicted_labels))
# %%

# %% Get ROC
#logit_roc_auc = roc_auc_score(y_val, model.predict(coefv_flat))
#fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(coefv_flat)[:,1])
#plt.figure()
#plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
#plt.plot([0, 1], [0, 1],'r--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic')
#plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
#plt.show()

