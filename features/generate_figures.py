# %%
import numpy as np
import pandas as pd
import sys
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml')
sys.path.append('/projectnb/cs542/wchapman/seizure_prediction_ml/features')

import utils
import pdc_dtf

# %%
pat = 3
df = pd.read_pickle('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s' +str(pat) +'_coefs')
npz = np.load('/projectnb/cs542/wchapman/seizure_prediction_ml/features/fits/s2_40Hz.npz')
dfs = utils.read_frames()
dfs = dfs[(dfs['pat']==pat) & (dfs['train']==1)]
y = dfs['class'].values

X1 = npz['X'][0]
X2 = npz['X'][2356]

coef = np.zeros((df.shape[0],
                 df.loc[0]['coef'][0].shape[0],
                 df.loc[0]['coef'][0].shape[1],
                 df.loc[0]['coef'][0].shape[2]))

for i in range(0, df.shape[0]):
        coef[i] = df.loc[i].coef[0]

# %% BIC
import matplotlib.pyplot as plt
%matplotlib auto

[pic1, bic1] = pdc_dtf.compute_order(X1.transpose(), p_max=15)
[pic2, bic2] = pdc_dtf.compute_order(X2.transpose(), p_max=15)

bic1 = bic1[1:-1]
bic2 = bic2[1:-1]

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 15}

plt.rc('font', **font)

bn1 = bic1 - bic1.min()
bn1 = bn1 / bn1.max()

bn2 = bic2 - bic2.min()
bn2 = bn2 / bn2.max()

plt.plot(bn1, linewidth=7)
plt.plot(bn2, linewidth=7)

plt.ylabel('Normed BIC')
plt.xlabel('Model Order')
plt.legend(['preictal','interictal'])
#plt.savefig('bic.png')

# %%


# %%
t1 = dfs['image'].values[0]
t2 = dfs['image'].values[2356]

d1 = utils.load_file(t1)
d2 = utils.load_file(t2)

# %% Plotting example traces
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

plt.rc('font', **font)


t = np.linspace(0,1,400)

plt.subplot(2,1,1)
plt.cla()
plt.plot(t, d1[400:800,:])
plt.ylabel('Potential (mV)')
plt.xlim(0,1)

plt.subplot(2,1,2)
plt.cla()
plt.plot(t,d2[400:800,:])
plt.ylabel('Potential (mV)')
plt.xlabel('Time (s)')
plt.xlim(0,1)

plt.show()

# %% Testing model order on non-downsampled data
[pic1, bic1] = pdc_dtf.compute_order(d1.transpose(), p_max=40)
[pic2, bic2] = pdc_dtf.compute_order(d2.transpose(), p_max=40)

bic1n = bic1[1:-1]
bic2n = bic2[1:-1]

# %%
bn1 = bic1n - bic1n.min()
bn1 = bn1 / bn1.max()

bn2 = bic2n - bic2n.min()
bn2 = bn2 / bn2.max()

plt.plot(bn1, linewidth=7)
plt.plot(bn2, linewidth=7)

# %% Plotting coefficients

plt.subplot(5,2,1)
plt.imshow(coef[0][0], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,3)
plt.imshow(coef[0][1], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,5)
plt.imshow(coef[0][2], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,7)
plt.imshow(coef[0][3], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,9)
plt.imshow(coef[0][4], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

#
plt.subplot(5,2,2)
plt.imshow(coef[2356][0], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,4)
plt.imshow(coef[2356][1], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,6)
plt.imshow(coef[2356][2], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,8)
plt.imshow(coef[2356][3], cmap=plt.get_cmap('PiYG'))
plt.axis('off')

plt.subplot(5,2,10)
plt.imshow(coef[2356][4], cmap=plt.get_cmap('PiYG'))
plt.axis('off')


#top=1.0,
#bottom=0.0,
#left=0.02,
#right=0.98,
#hspace=0.0,
#wspace=0.08


# %% Get number of trials for each subject
df = utils.read_frames()
print(np.all(((df.pat==1), (df.train==1), df['class'].values==0), axis=0).sum())
print(np.all(((df.pat==1), (df.train==1), df['class'].values==1), axis=0).sum())

print(df['class'][np.all(((df.pat==1), (df.train==1)), axis=0)].mean())

print(np.all(((df.pat==1), (df.train==0)), axis=0).sum())

# subj 2
print(np.all(((df.pat==2), (df.train==1), df['class'].values==0), axis=0).sum())
print(np.all(((df.pat==2), (df.train==1), df['class'].values==1), axis=0).sum())

print(df['class'][np.all(((df.pat==2), (df.train==1)), axis=0)].mean())


print(np.all(((df.pat==2), (df.train==0)), axis=0).sum())

# subj 3
print(np.all(((df.pat==3), (df.train==1), df['class'].values==0), axis=0).sum())
print(np.all(((df.pat==3), (df.train==1), df['class'].values==1), axis=0).sum())

print(df['class'][np.all(((df.pat==3), (df.train==1)), axis=0)].mean())


print(np.all(((df.pat==3), (df.train==0)), axis=0).sum())