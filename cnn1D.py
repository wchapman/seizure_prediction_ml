# %% imports
import keras
import keras.layers as layers
import utils


# %% Get example of data size, for determining layer sizes
downsample = 10
sampleProp = 1

X_train, y_train, X_val, y_val, X_test = utils.gen_dataset(1, sampleProp=100, downsample=downsample, freqs=None, coh=False)

nb_features = X_train[0].shape[1]
nb_out = 1
sequence_length = X_train[0].shape[0]

# %% create model
model = keras.Sequential()

# model.add(Reshape((sequence_length, nb_features), input_shape=(input_shape,)))
model.add(layers.Conv1D(
            filters = 100,
            kernel_size = 10,
            activation ='relu',
            input_shape = (sequence_length, nb_features)))
# model.add(layers.MaxPooling1D(pool_size = 3))
model.add(layers.Conv1D(
            filters = 100, 
            kernel_size = 10, 
            activation='relu'))
model.add(layers.MaxPooling1D(pool_size = 3))
model.add(layers.Conv1D(
            filters = 160,
            kernel_size = 10, 
            activation='relu'))
model.add(layers.Conv1D(
            filters = 160, 
            kernel_size = 10, 
            activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dropout(rate = 0.5))
model.add(layers.Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# %% Pass into fitting/eval method
utils.fit_eval_model(model, Name='baseline', downsample=downsample, sampleProp=sampleProp, pats=[1])
