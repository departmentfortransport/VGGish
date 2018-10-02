from numpy import genfromtxt
np.set_printoptions(suppress=True)

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from matplotlib import pyplot as plt

X_train = genfromtxt("data/embs/X_train.csv", delimiter=',')
y_train = genfromtxt("data/embs/y_train.csv", delimiter=',')
X_train.shape
y_train.shape

X_eval = genfromtxt("data/embs/X_eval.csv", delimiter=',')
y_eval = genfromtxt("data/embs/y_eval.csv", delimiter=',')
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
    EarlyStopping(monitor='val_loss', verbose=2, patience=10),
    ]

history = model.fit(
     X_train,
     y_train,
     batch_size=batch_size,
     epochs=100,  # redundant with callbacks
     validation_data=(X_eval, y_eval),
     shuffle=True,
     callbacks=callbacks)

# In[9]:
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

# In[9]:
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
