# %%
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy

# %% Loading
fname = '/media/wchapman/Data/GoogleDrive/epieco/Pat1Train/Pat1Train_21_0.hdf5'
file = h5py.File(fname,'r')
dat = file.get('data/table').value

sn = np.zeros((dat.shape[0],1))
samples = np.zeros((dat.shape[0], 16))
for i in range(0, dat.shape[0]):
    sn[i] = dat[i][0]
    samples[i, :] = dat[i][3]


#Not sure what all these fields are for. Maybe come back to them
#dd = file.get('data/_i_table/index')
#('abounds', <HDF5 dataset "abounds": shape (160,), type "<i8">)
#('bounds', <HDF5 dataset "bounds": shape (1, 159), type "<i8">)
#('indices', <HDF5 dataset "indices": shape (1, 163840), type "<u4">)
#('indicesLR', <HDF5 dataset "indicesLR": shape (163840,), type "<u4">)
#('mbounds', <HDF5 dataset "mbounds": shape (160,), type "<i8">)
#('mranges', <HDF5 dataset "mranges": shape (1,), type "<i8">)
#('ranges', <HDF5 dataset "ranges": shape (1, 2), type "<i8">)
#('sorted', <HDF5 dataset "sorted": shape (1, 163840), type "<i8">)


# %% Get spectral responses
def butter_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scipy.signal.butter(order, [low, high],  btype='band')
    signal = scipy.signal.lfilter(b, a, data)
    hb = scipy.signal.hilbert(signal)
    amp = np.real(hb)
    ang = np.imag(hb)
    return signal, amp, ang


delta_s, delta_a, delta_p = butter_filter(samples, 0.1, 4, 400, 2)
theta_s, theta_a, theta_p = butter_filter(samples, 4, 8, 400, 2)
alpha_s, alpha_a, alpha_p = butter_filter(samples, 8, 12, 400, 2)
beta_s, beta_a, beta_p = butter_filter(samples, 12, 30, 400, 2)
gamma_s, gamma_a, gamma_p = butter_filter(samples, 30, 60, 400, 2)
