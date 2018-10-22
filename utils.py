"""
Helper functions for epilepsy prediction project.
"""

import numpy as np
import h5py
import scipy
import scipy.signal


def load_file(fname):
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

    return samples


def butter_filter(data, lowcut, highcut, fs, order=2):
    """
    Performs time-frequency filtering on data
    :param data: numpy array of data to filter
    :param lowcut: lowpass end of butterworth filter (Hz)
    :param highcut: highpass end (Hz)
    :param fs: Sampling rate of data (Hz)
    :param order: Filter order (default: 2)
    :return: filtered signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high],  btype='band')
    signal = scipy.signal.lfilter(b, a, data)
    # hb = scipy.signal.hilbert(signal)
    # amp = np.real(hb)
    # ang = np.imag(hb)
    return signal


def spectral_responses(data):
    """
    :param data:
    :return:
    """
    delta = butter_filter(data, 0.1, 4, 400, 2)
    theta = butter_filter(data, 4, 8, 400, 2)
    alpha= butter_filter(data, 8, 12, 400, 2)
    beta= butter_filter(data, 12, 30, 400, 2)
    gamma = butter_filter(data, 30, 60, 400, 2)

    return delta, theta, alpha, beta, gamma





# Not sure what all these fields are for. Maybe come back to them
# dd = file.get('data/_i_table/index')
# ('abounds', <HDF5 dataset "abounds": shape (160,), type "<i8">)
# ('bounds', <HDF5 dataset "bounds": shape (1, 159), type "<i8">)
# ('indices', <HDF5 dataset "indices": shape (1, 163840), type "<u4">)
# ('indicesLR', <HDF5 dataset "indicesLR": shape (163840,), type "<u4">)
# ('mbounds', <HDF5 dataset "mbounds": shape (160,), type "<i8">)
# ('mranges', <HDF5 dataset "mranges": shape (1,), type "<i8">)
# ('ranges', <HDF5 dataset "ranges": shape (1, 2), type "<i8">)
# ('sorted', <HDF5 dataset "sorted": shape (1, 163840), type "<i8">)