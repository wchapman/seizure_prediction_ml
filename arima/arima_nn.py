# %%
import numpy as np
import pandas as pd
import sys
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml/arima')
import pdc_dtf

import utils

# %%
pat = 1
df = pd.read_pickle('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' +str(pat) +'_coefs')
#npz = np.load('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s2_40Hz.npz')
dfs = utils.read_frames()
dfs = dfs[(dfs['pat']==pat) & (dfs['train']==1)]
y = dfs['class'].values

# %%
# unpack the variables
coef = np.zeros((df.shape[0], 
                 df.loc[0]['coef'][0].shape[0], 
                 df.loc[0]['coef'][0].shape[1],
                 df.loc[0]['coef'][0].shape[2]))

coef_block = np.zeros((df.shape[0], 
                       df.loc[0]['coef_block'][0].shape[0], 
                       df.loc[0]['coef_block'][0].shape[1],
                       df.loc[0]['coef_block'][0].shape[2],
                       df.loc[0]['coef_block'][0].shape[3]))

coef_flat = np.zeros((coef.shape[0],coef.shape[1]*coef.shape[2]*coef.shape[3]))
coef_flat_block = np.zeros((coef.shape[0], coef.shape[1], coef.shape[2]*coef.shape[3]))
y_blocks = np.zeros((coef.shape[0], coef.shape[1]))

for i in range(0, df.shape[0]):
    coef[i] = df.loc[i].coef[0]
    coef_block[i] = df.loc[i].coef_block[0]
    
    coef_flat[i] = coef[i].flatten()
    
    for k in range(0, coef[i].shape[0]):
        coef_flat_block[i][k] = coef[i][k].flatten()
        y_blocks[i] = y[i]
        
# %%imports
from sklearn.model_selection import GridSearchCV
import keras
from keras.models import Sequential
import keras.layers as layers
from keras.wrappers.scikit_learn import KerasClassifier
import keras.regularizers as regularizers

# %%
def create_model(dropout=0., nn = [1], l1=0, l2=0):
    model = Sequential()

    for i in range(0, nn.__len__()):
        if i == 0:
            model.add(layers.Dense(nn[i], 
                      kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                      input_dim=1280))
        else:
            model.add(layers.Dense(nn[i], 
                      kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2)))
        
        model.add(layers.Dropout(dropout))
        
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# %%
l1 = np.append(np.power(10., np.arange(-5, -1, 2)), 0)
l2 = np.append(np.power(10., np.arange(-5, -1, 2)), 0)
epochs = [10]
batch_size = [4, 32, 64]
dropout = np.arange(0, 0.3, 0.05)

nn = [
    [32],
    [64],
    [32, 64],
    [64, 32],
    [32, 64, 32]
]

print(l1.__len__()*l2.__len__()*epochs.__len__()*batch_size.__len__()*dropout.__len__()*nn.__len__())

params = dict(l1=l1, l2=l2, epochs=epochs, batch_size=batch_size, nn=nn)

model = KerasClassifier(build_fn=create_model, verbose=0)
grid = GridSearchCV(estimator=model, param_grid=params, cv=10, verbose=10)
grid_result = grid.fit(coef_flat, y)

import pickle
pickle.dump(grid_result, open( "save.p", "wb" ) )

# %%
from sklearn.model_selection import train_test_split

model = Sequential()
model.add(layers.Dense(32,
                      #kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                      input_dim=1280))

model.add(layers.Dropout(0.2))

model.add(layers.Dense(64))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(32))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

x_train, x_test, y_train, y_test = train_test_split(coef_flat, y, stratify=y)

model.fit(x_train, y_train, batch_size=32, epochs=1000, verbose=0)

# %% Get ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(y_train, model.predict_proba(x_train))
print(logit_roc_auc)
logit_roc_auc = roc_auc_score(y_test, model.predict_proba(x_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(x_test))
plt.figure()
plt.plot(fpr, tpr, label='NN (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()