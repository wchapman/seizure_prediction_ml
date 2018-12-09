# %% imports
import keras
import keras.layers as layers
import utils
import numpy as np

# %% Load pre-downsampled dataset
pat=2
npz = np.load('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s2_40Hz.npz')
dfs = utils.read_frames()
dfs = dfs[(dfs['pat']==pat) & (dfs['train']==1)]
X = npz['X']
y = dfs['class'].values


# %% Split for training/test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

# %% create model

nb_features = X_train[0].shape[1]
nb_out = 1
NFilters=40
sequence_length = X_train[0].shape[0]


model = keras.models.Sequential()

model.add(layers.LSTM(
        input_shape=(sequence_length, nb_features),
        units=100, return_sequences=False))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
model.fit(X_train, y_train, batch_size=32, epochs=10)


# %% Pass into fitting/eval method

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

predi = model.predict_proba(X_train)

logit_roc_auc = roc_auc_score(y_train, predi)
print(logit_roc_auc)

predi = model.predict_proba(X_test)
logit_roc_auc = roc_auc_score(y_test, predi)
fpr, tpr, thresholds = roc_curve(y_test, predi)
plt.figure()
plt.plot(fpr, tpr, label='LSTM (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()
