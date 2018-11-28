"""
Uses ARIMA to fit coefficients for the entire session as well as 30s intervals. Saves these to disk for later use in classification tasks.
"""
# %% Setup
import numpy as np
import pandas as pd
import pdc_dtf
import sys
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
import utils

# %% configs
subject = 3
model_order = 5      # 10 = 10Hz
blockSize =   40*30  # 40 = 1 second

# %% Import data
X, y, lst = utils.load_all_subj(subject, sampleProp=1, downsample=10, Freqs=False, coh=False)

# %% Fit coefs for entire session
coef = np.zeros((X.shape[0], model_order, 16, 16))

for i in range(0, X.shape[0]):
    try:
        coef[i], noise = pdc_dtf.mvar_fit(X[i, :, :].transpose(), model_order)
    except:
        pass

# %% Get coef for each block
coef_block = np.zeros((X.shape[0], int(X.shape[1]/blockSize), model_order, 16, 16))
for i in range(0, X.shape[0]):              # for each trial
    for k in range(0, coef_block.shape[1]): # for each block
        try:
            inds = np.arange(k*blockSize, (k+1)*blockSize)
            coef_block[i][k], noise = pdc_dtf.mvar_fit(X[i][inds][:].transpose(), model_order)
        except: # missing data in this block
            coef_block[i][k] = np.zeros(coef_block[0][0].shape)

# %% Convert to dataframe
df = pd.DataFrame(data={'files': lst})
df['coef'] = ""
df['coef_block'] = ""
for i in range(0, coef.shape[0]):
    df.loc[i]['coef'] =  [coef[i,:,:,:]]
    df.loc[i]['coef_block'] = [coef_block[i,:,:,:,:]]

# %% Save to disk
np.savez('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' + str(subject) +'_40Hz', X=X, y=y, lst=lst) 
df.to_pickle('/projectnb/cs542/wchapman/seizure_prediction_ml/arima/fits/s' + str(subject) + '_coefs')