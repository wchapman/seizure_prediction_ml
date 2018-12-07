"""
Helper functions for epilepsy prediction project.
"""

import numpy as np
import h5py
import scipy
import scipy.signal
import skimage.measure
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import sklearn.metrics
from tensorflow import keras
import sklearn.preprocessing as preprocessing

# %%
def fit_eval_model(model_orig, pats=[1,2,3], Name=None, epochs=10, batch_size=10, sampleProp=1, downsample=10, freqs=None, coh=False):
    """
    Deals with all loading/pre-proc/training/validation for the project. Simply pass in a model, some optional
    parameters, and this will deal with the rest.

    :param model: A fully specified Keras model
    :param Name: string of model name (default: datestr)

    :return: None. Saves model and performance information
    """

    for pat in pats:
        model = keras.models.clone_model(model_orig)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        X_train, y_train, X_val, y_val, X_test = gen_dataset(pat, sampleProp=sampleProp, downsample=downsample, freqs=freqs, coh=coh)

        model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

        # training
        loss_train, acc_train = model.evaluate(X_train, y_train)
        pred_train = model.predict(X_train)
        auc_train = sklearn.metrics.roc_auc_score(y_train, pred_train)

        # cross validation
        loss_val, acc_val = model.evaluate(X_val, y_val)
        pred_val = model.predict(X_val)
        auc_val = sklearn.metrics.roc_auc_score(y_val, pred_val)

        model.save('../outputs/' + Name + '_' + str(pat) + '.h5')
        with open('../outputs/' + Name + '.txt', 'w') as f:
            f.write("loss_train: %s\n" % loss_train)
            f.write("acc_train: %s\n" % acc_train)
            f.write("auc_train: %s\n" % auc_train)
            f.write("loss_val: %s\n" % loss_val)
            f.write("acc_val: %s\n" % acc_val)
            f.write("auc_val: %s\n" % auc_val)


# %%
def gen_dataset(pat, sampleProp=1, downsample=10, Freqs=False, coh=False):
    """
    Generates numpy arrays that can be directly fed into a Keras or sklean model.

    :param pat: Which patient
    :param sampleProp: Take only every nth trial, for quick training
    :param downsample: Downsample by some factor
    :param freqs: Calculates power in these 1Hz bands and passes into model   #TODO
    :param coh: Calculates coherence between channels, and passes into model  #TODO

    :return: (X_train, y_train, X_val, y_val, X_test)
    """

    #  Load data
    df = read_frames()
    lst = df['image'][(np.logical_and(np.equal(df['pat'], pat), df['train']))]  # list of files
    labels = df['class'][(np.logical_and(np.equal(df['pat'], pat), df['train']))]

    lst = lst.values[np.arange(0, len(labels), sampleProp)]
    labels = labels.values[np.arange(0, len(labels), sampleProp)]

    dat = []

    for fn in lst:
        ld = load_file(fn, downsample=downsample, coh=coh, Freqs=Freqs)
        #TODO: append preproc features here
        dat.append(ld)

    dat = np.asarray(dat)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(dat, labels, test_size=0.3)

    # test data
    lst = df['image'][(np.logical_and(np.equal(df['pat'], pat), df['test']))]
    dat = []
    for fn in lst:
        ld = load_file(fn, downsample=downsample)
        #TODO: append preproc features here
        dat.append(ld)
    X_test = np.asarray(dat)

    return X_train, y_train, X_val, y_val, X_test

# %%
def load_all_subj(pat, sampleProp=1, downsample=10, Freqs=False, coh=False):
    """
    Generates numpy arrays that can be directly fed into a Keras or sklean model.

    :param pat: Which patient
    :param sampleProp: Take only every nth trial, for quick training
    :param downsample: Downsample by some factor
    :param freqs: Calculates power in these 1Hz bands and passes into model   #TODO
    :param coh: Calculates coherence between channels, and passes into model  #TODO

    :return: (X_train, y_train, X_val, y_val, X_test)
    """

    #  Load data
    df = read_frames()
    lst = df['image'][(np.logical_and(np.equal(df['pat'], pat), df['train']))]  # list of files
    labels = df['class'][(np.logical_and(np.equal(df['pat'], pat), df['train']))]

    lst = lst.values[np.arange(0, len(labels), sampleProp)]
    labels = labels.values[np.arange(0, len(labels), sampleProp)]

    dat = []

    for fn in lst:
        ld = load_file(fn, downsample=downsample, coh=coh, Freqs=Freqs)
        #TODO: append preproc features here
        dat.append(ld)

    dat = np.asarray(dat)
    return dat, labels, lst

# %%
def load_file(fname, downsample=1, coh=False, Freqs=False):
    """
    loadFile(fullFileName.hd5)
    returns: 10 minutes of data in a (240000, 16) numpy array
    """

    file = h5py.File(fname, 'r')
    dat = file.get('data/table').value

    sn = np.zeros((dat.shape[0], 1))
    samples = np.zeros((dat.shape[0], 16))
    for i in range(0, dat.shape[0]):
        sn[i] = dat[i][0]
        samples[i, :] = dat[i][3]

    if coh:
        pass # TODO: Implement me

    if Freqs:
        samples = spectral_responses(samples)

    samples = skimage.measure.block_reduce(samples, block_size=(downsample, 1), func=np.mean)

    return samples


# %%
def wavelet(data):
    from scipy import signal
    from sklearn import preprocessing
    fs = 400
    sig = signal.cwt(data, signal.ricker, (fs/2, fs/6, fs/10, fs/21, fs/45))
    sig = np.apply_along_axis(sklearn.preprocessing.scale, 1, sig)
    sig = np.swapaxes(sig, 0, 1)
    return sig
    

# %%
def spectral_responses(data):
    """
    :param data:
    :return:
    """
    data_ap = data
    for chan in np.arange(0, data.shape[1]):
        data_ap = np.append(data_ap,wavelet(data[:, chan]), axis=1)
    return data_ap

# %%
def read_frames(fname=None):
    if fname is None:
        fname = '/projectnb/cs542/wchapman/'
    f1 = pd.read_csv(fname + 'contest_train_data_labels.csv')
    f2 = pd.read_csv(fname + 'contest_test_data_labels_public.csv')
    f2 = f2.drop(columns=['usage'])

    f = pd.concat([f1, f2])

    f['pat'] = f['image'].str.contains('Pat1')*1 + f['image'].str.contains('Pat2')*2 + f['image'].str.contains('Pat3')*3
    f['test'] = np.isnan(f['class']) + 0
    f['train'] = np.equal(f['test'], 0) + 0
    f['image'] = fname + 'data/' + f['image'] + '.hdf5'
    f = f.reindex()
    return f

