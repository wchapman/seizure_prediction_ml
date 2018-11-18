"""
Uses ARIMA to extract coefficients, and feeds those into a simple CNN for classification.
"""
# %%
import numpy as np
import utils
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd
import sklearn

# %% Import data
X_train, y_train, X_val, y_val, X_test = utils.gen_dataset(1, sampleProp=1, downsample=10, freqs=None, coh=False)
t = np.arange(0, 60*10, 1/40)


# %%
import pdc_dtf
coef = np.zeros((X_train.shape[0], 5, 16, 16))

for i in range(0, X_train.shape[0]):
    print(i)
    try:
        coef[i], noise = pdc_dtf.mvar_fit(X_train[i, :, :].transpose(), 5)
    except:
        pass

 # %%
coef_flat = np.zeros((628, 1280))
for i in range(0, 628):
    coef_flat[i,:] = coef[i].flatten()

m0 = coef_flat[y_train == 0].mean(axis=0)
s0 = coef_flat[y_train == 0].std(axis=0)

m1 = coef_flat[y_train == 1].mean(axis=0)
s1 = coef_flat[y_train == 1].std(axis=0)

x = np.arange(0, 1280)
plt.fill_between(x, m0-s0, m0+s0)
plt.fill_between(x, m1-s1, m1+s1, facecolor='red')

# %% Logistic regression setup
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
model = LogisticRegression()
model.fit(coef_flat, y_train)


# %%
coef_val = np.zeros((X_val.shape[0], 5, 16, 16))
for i in range(0, X_val.shape[0]):
    try:
        coef_val[i], noise = pdc_dtf.mvar_fit(X_val[i,:,:].transpose(), 5)
    except:
        pass

coefv_flat = np.zeros((coef_val.shape[0],1280))
for i in range(0, coef_val.shape[0]):
    coefv_flat[i,:] = coef_val[i].flatten()

# %%
y_pred = model.predict(coefv_flat)
print(model.score(coefv_flat, y_val))
# Accuracy of 0.78 on subject 1 ... not horrible

# %% Get ROC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_val, model.predict(coefv_flat))
fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(coefv_flat)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# %% Model Fit
import keras
import keras.layers as layers



NFilters = 40
nb_features = coef_flat.shape[1]
nb_out = 1
batch_size = 10
epochs = 100

model = keras.models.Sequential()

#model.add(layers.Conv1D(filters=NFilters,
#                        kernel_size=nb_features, padding='same',
#                        input_shape=(1, nb_features)))

#model.add(layers.Dropout(0.2))

model.add(layers.Dense(36, activation='relu', input_dim=1280))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=nb_out, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(coef_flat, y_train, batch_size=batch_size, epochs=epochs)

# %% Eval training
loss_train, acc_train = model.evaluate(coef_flat, y_train)
pred_train = model.predict(coef_flat)
auc_train = sklearn.metrics.roc_auc_score(y_train, pred_train)
print("%s\n" %auc_train)

# %% Eval validation
loss_val, acc_val = model.evaluate(coefv_flat, y_val)
pred_val = model.predict(coefv_flat)
auc_val = sklearn.metrics.roc_auc_score(y_val, pred_val)
print("%s\n" %auc_val)
