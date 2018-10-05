import numpy as np
from numpy import genfromtxt
np.set_printoptions(suppress=True)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

# TODO: put the VGGish model as the bottom layers of KEras model rather than
# have two separate processes
# TODO: try with embeddings per second, rather than the sum over the whole clip
# TODO: address the overfitting - dropout, regularization

rng_state = np.random.get_state()
rng_state

# need to shuffle as positive and negative classes are ordered
X_train = genfromtxt("data/embs/X_unbal.csv", delimiter=',')
np.random.shuffle(X_train)
y_train = genfromtxt("data/embs/y_unbal.csv", delimiter=',')
np.random.set_state(rng_state)
np.random.shuffle(y_train)
X_train.shape
y_train.shape

X_eval = genfromtxt("data/embs/X_eval.csv", delimiter=',')
np.random.set_state(rng_state)
np.random.shuffle(X_eval)
y_eval = genfromtxt("data/embs/y_eval.csv", delimiter=',')
np.random.set_state(rng_state)
np.random.shuffle(y_eval)

X_train.shape
y_train.shape
X_eval.shape
y_eval.shape

layer_dims = [32, 16, 8 ,1]
batch_size = 32

model = Sequential()
model.add(Dense(layer_dims[0], input_dim=128, activation='relu'))
model.add(Dense(layer_dims[1], activation='relu'))
model.add(Dense(layer_dims[2], activation='relu'))
model.add(Dense(layer_dims[3], activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


callbacks = [
    EarlyStopping(monitor='val_loss', verbose=2, patience=5),
    ]

history = model.fit(
     X_train,
     y_train,
     batch_size=batch_size,
     epochs=1000,  # redundant with callbacks
     validation_data=(X_eval, y_eval),
     shuffle=True,
     callbacks=callbacks)

# In[1]:
# Run all in one go to get the graph (Option+Shift+Enter)

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(len(loss))

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss,  'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# In[2]:
# Run all in one go to get the graph (Option+Shift+Enter)
history_dict = history.history

acc = history_dict['acc']
val_acc = history_dict['val_acc']

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc,  'b', label='Validation acc')
plt.title('Training and validation acc')
plt.legend()
plt.show()
