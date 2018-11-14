# %%
import numpy as np
import utils
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

# %%
X_train, y_train, X_val, y_val, X_test = utils.gen_dataset(1, sampleProp=1, downsample=10, freqs=None, coh=False)
t = np.arange(0, 60*10, 1/40)

# %%
dat = X_train[1, 0:40, :]
from statsmodels.tsa.statespace.varmax import VARMAX

maxOrder = 10
bic = np.zeros((maxOrder, 1))
aic = np.zeros((maxOrder, 1))
fts = list()
for mo in range(0, maxOrder):
    print(mo)
    try:
        model = VARMAX(dat, order=(mo, 0))
        ft = model.fit()
        bic[mo] = ft.bic
        aic[mo] = ft.aic
        fts.append(ft)
    except:
        aic[mo] = np.nan
        bic[mo] = np.nan
        fts.append(np.nan)

# %%
import pdc_dtf
[est, bic] = pdc_dtf.compute_order(X_train[0, :, :].transpose(), 10)
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