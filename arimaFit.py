# %%
import numpy as np
import utils
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import pandas as pd

# %%
X_train, y_train, X_val, y_val, X_test = utils.gen_dataset(1, sampleProp=1, downsample=10, freqs=None, coh=False)

# %%
model = ARIMA(endog=X_train[0,:,0], order=(10,1,0))
ft = model.fit(disp=0)
