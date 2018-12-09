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

# %% Load coefficients
n_blocks=20
pat=2
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
# %%
dropout = 0.2
l1 = 0.
l2 = 0.
sequence_length = 20
nb_features = 1280

model = keras.Sequential()
model.add(layers.TimeDistributed(
            layers.Dense(32, input_dim=1280)))

model.add(layers.Dropout(dropout))

model.add(layers.TimeDistributed(
        layers.Dense(64)))

model.add(layers.Dropout(dropout))


model.add(layers.LSTM(units=32, return_sequences=True,
                      kernel_regularizer=keras.regularizers.l1_l2(l1, l2)))

model.add(layers.LSTM(units=32, return_sequences=False,
                      kernel_regularizer=keras.regularizers.l1_l2(l1, l2)))

model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=10)



# %% Get ROC

logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
print(logit_roc_auc)
logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
plt.plot(fpr, tpr, label='Pat%d (AUC = %0.2f)' %(pat, logit_roc_auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.title('NN: 0.87')
plt.legend(loc="lower right")
#plt.savefig('final_ROC')