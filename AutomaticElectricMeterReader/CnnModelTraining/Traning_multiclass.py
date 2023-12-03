import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle

print("<<<<<<<<<<<<  v e r s i o n 1.0 >>>>>>>>>>>>>>>>>>>")
pickle_in = open("X_multiclass.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y_multiclass.pickle","rb")
y = pickle.load(pickle_in)

print(X.shape)
y=np.array(y)
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=1))



model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

print("<<<<<<<<<<<<<<<<<<<<<< M O D E L   S U M M A R Y  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(model.summary())
N = 700
H =model.fit(X_train, y_train, batch_size=32, epochs=N, validation_split=0.1)

plt.style.use("ggplot")
plt.figure()

plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy ")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig('graph3last_3layer.png')
plt.show()
model.save("last_3layer.h5")

#testing of model
print("###########################################################################")
print("Evaluation on testing data :")
result=model.evaluate(X_test, y_test, verbose = 0)
print(f'Test loss: {result[0]} / Test accuracy: {result[1]}')
