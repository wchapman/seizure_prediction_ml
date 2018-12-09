# %% import

import numpy as np
import sys
import scipy

sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml/features')
import utils
import pdc_dtf

# %% define functions
def hurst(ts):
    """Returns the Hurst Exponent of the time series vector ts"""
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0


def norm(var):
    nvar = var - var.mean()
    nvar = nvar / nvar.std()
    return nvar


def coh(block):
    vals = list()
    for x in range(0, 16):
        for y in range(0, 16):
            if not x == y:
                f, val = scipy.signal.coherence(block[:, x], block[:, y], fs=40.0, nperseg=20)
                vals.append(val)
        return np.asarray(vals).flatten()


def block_fets(block):
    block_freq = utils.spectral_responses(block)

    f1 = norm(np.apply_along_axis(hurst, 0, block))
    f2 = norm(block.std(axis=0))
    f3 = norm(block_freq.mean(axis=0))
    f4 = norm(scipy.stats.skew(block_freq, axis=0))
    f5 = norm(scipy.stats.kurtosis(block_freq, axis=0))
    f6 = norm(np.power(block, 2).mean(axis=0))
    f7 = coh(block)
    f8, noise = pdc_dtf.mvar_fit(block.transpose(), 5)

    standard_fets = np.concatenate((f1, f2, f3, f4, f5, f6, f7))
    ar_fets = f8.flatten()

    return standard_fets, ar_fets

# %% Process
pats = (1,2,3)
block_sizes = (40*30, 40*60, 40*600)

for pat in pats:
    npz = np.load('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s' + str(pat) + '_40Hz.npz')
    X = npz['X']

    for block_size in block_sizes:

        n_blocks = int(X.shape[1]/block_size)
        standard_fets = np.zeros((X.shape[0], n_blocks, 501))
        ar_fets = np.zeros((X.shape[0], n_blocks, 1280))

        #% begin
        for i in range(0, X.shape[0]):              # for each trial
            for k in range(0, n_blocks): # for each block
                try:
                    inds = np.arange(k*block_size, (k+1)*block_size)
                    sf, af = block_fets(X[i,inds,:])
                except: # missing data in this block
                    sf = np.zeros(standard_fets.shape[2])
                    af = np.zeros(ar_fets.shape[2])
                standard_fets[i, k, :] = sf
                ar_fets[i, k, :] = af

        np.savez('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/fets_'+ str(pat) +'_' + str(n_blocks) , standard_fets=standard_fets, ar_fets=ar_fets)