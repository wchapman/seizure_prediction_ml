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
model = keras.models.Sequential()

model.add(layers.LSTM(
           input_shape=(sequence_length, nb_features),
           units = 100,
           return_sequences=True))
#model.add(layers.LSTM(
#           input_shape=(sequence_length, nb_features),
#           units = 100,
#           return_sequences=True))

model.add(layers.Dropout(0.2))
model.add(layers.LSTM(
           units = 50,
           return_sequences=False))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% Pass into fitting/eval method
utils.fit_eval_model(model, Name='baseline', downsample=downsample, sampleProp=sampleProp, pats=[1])
