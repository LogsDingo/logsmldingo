
























*****************************************************************************
Practical 1 Matrix multiplication
*****************************************************************************
import tensorflow as tf
print("Matrix Multiplication Demo")
x=tf.constant([1,2,3,4,5,6],shape=[2,3])
print(x)
y=tf.constant([7,8,9,10,11,12],shape=[3,2])
print(y)
z=tf.matmul(x,y)
print("Product:",z)
e_matrix_A=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,
name="matrixA")
print("Matrix A:\n{}\n\n".format(e_matrix_A))
eigen_values_A,eigen_vectors_A=tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors:\n{}\n\nEigen
Values:\n{}\n".format(eigen_vectors_A,eigen_values_A))
*****************************************************************************
Practical 2 XOR
*****************************************************************************
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
model=Sequential()
model.add(Dense(units=2,activation='relu',input_dim=2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc
uracy'])
print(model.summary())
print(model.get_weights())
X=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y=np.array([0.,1.,1.,0.])
model.fit(X,Y,epochs=1000,batch_size=4)
print(model.get_weights())
print(model.predict(X,batch_size=4))
*****************************************************************************
Practical 3 Binary Classification
*****************************************************************************
from numpy import loadtxt
from keras.models import Sequential 
from keras.layers import Dense
dataset =loadtxt('C:/Users/rohan/PycharmProjects/DL_PRACTICAL/pima- indians-diabetes.csv', delimiter=',')
X = dataset[:, 0:8] Y = dataset[:, 8]
print(X) print(Y)
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu')) model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
accuracy=model.evaluate(X,Y)
print("Accuracy of model is", (accuracy*100))
prediction = model.predict(X, batch_size=4)
print(prediction)
exec("for i in range(5):print(X[i].tolist(),prediction[i], Y[i])")
*****************************************************************************
Practical 4A predicting class
*****************************************************************************
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
X,Y=make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)
scalar=MinMaxScaler()
scalar.fit(X)
X=scalar.transform(X)
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X,Y,epochs=500)
Xnew,Yreal=make_blobs(n_samples=3,centers=2,n_features=2,random_state=1
)
Xnew=scalar.transform(Xnew)
Ynew=model.predict_classes(Xnew)
for i in range(len(Xnew)):
print("X=%s,Predicted=%s,Desired=%s"%(Xnew[i],Ynew[i],Yreal[i]))
*****************************************************************************
Practical 4B Probability of a Class
*****************************************************************************
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
X,Y=make_blobs(n_samples=100,centers=2,n_features=2,random_state=1)
scalar=MinMaxScaler()
scalar.fit(X)
X=scalar.transform(X)
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(X,Y,epochs=500)
Xnew,Yreal=make_blobs(n_samples=3,centers=2,n_features=2,random_state=1
)
Xnew=scalar.transform(Xnew)
Yclass=model.predict(Xnew)
Ynew=model.predict(Xnew) # Use predict to get predicted probabilities
for i in range(len(Xnew)):
print("X=%s,Predicted_probability=%s,Predicted_class=%s"%(Xnew[i],Ynew
[i],Yclass[i]))
*****************************************************************************
Practical 4C Linear Regression
*****************************************************************************
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
X,Y=make_regression(n_samples=100,n_features=2,noise=0.1,random_state=1
)
scalarX,scalarY=MinMaxScaler(),MinMaxScaler()
scalarX.fit(X)
scalarY.fit(Y.reshape(100,1))
X=scalarX.transform(X)
Y=scalarY.transform(Y.reshape(100,1))
model=Sequential()
model.add(Dense(4,input_dim=2,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mse',optimizer='adam')
model.fit(X,Y,epochs=1000,verbose=0)
Xnew,a=make_regression(n_samples=3,n_features=2,noise=0.1,random_state=
1)
Xnew=scalarX.transform(Xnew)
Ynew=model.predict(Xnew)
for i in range(len(Xnew)):
print("X=%s,Predicted=%s"%(Xnew[i],Ynew[i]))
*****************************************************************************
Practical 5A KFold Cross Validation
*****************************************************************************
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.models import
Sequentialfrom tensorflow.keras.layers
import Dense
X=np.random.rand(1000,
10)y = np.sum(X,
axis=1)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)eval_metrics = []
for train_index, test_index in kfold.split(X):
# Split data into training and testing sets X_train, X_test =
X[train_index], X[test_index]y_train, y_test = y[train_index],
y[test_index]
model = Sequential()
model.add(Dense(64, activation='relu',
input_dim=10))model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
eval_metrics.append(model.evaluate(X_test, y_test))
print("Average evaluation metrics:")
print("Loss:", np.mean([m[0] for m in
eval_metrics]))print("MAE:", np.mean([m[1] for
m in eval_metrics]))
*****************************************************************************
Practical 5B Feedforward Multiclass Classification
*****************************************************************************
pip install scikeras
from scikeras.wrappers import KerasClassifier
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('/content/flowers.csv', header=0)
print(df.head())
X = df.iloc[:, :-1].astype(float)
y = df.iloc[:, -1]
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
print(encoded_y)
dummy_Y = to_categorical(encoded_y)
print(dummy_Y)
def baseline_model():
model = Sequential()
model.add(Dense(8, input_dim=X.shape[1], activation='relu'))
model.add(Dense(dummy_Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
metrics=['accuracy'])
return model
estimator = baseline_model()
estimator.fit(X, dummy_Y, epochs=100, shuffle=True)
action = estimator.predict(X)
for i in range(25):
print(dummy_Y[i])
print('^^^^^^^^^^^^^^^^^^^^^^')
for i in range(25):
print(action[i])
*****************************************************************************
Practical 6A Implement Regularization
*****************************************************************************
from matplotlib import pyplot
from sklearn.datasets import make_moons from keras.models import Sequential from
keras.layers import Dense
X,Y=make_moons(n_samples=100,noise=0.2,random_state=1) n_train=30
trainX,testX=X[:n_train,:],X[n_train:] trainY,testY=Y[:n_train],Y[n_train:]
model=Sequential()
model.add(Dense(500,input_dim=2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc uracy'])
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=40 00)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test') pyplot.legend()
*****************************************************************************
Practical 6B L2
*****************************************************************************
from matplotlib import pyplot
from sklearn.datasets import make_moons from keras.models import Sequential from
keras.layers import Dense
from keras.regularizers import l2
X,Y=make_moons(n_samples=100,noise=0.2,random_state=1) n_train=30
trainX,testX=X[:n_train,:],X[n_train:] trainY,testY=Y[:n_train],Y[n_train:]
model=Sequential()
model.add(Dense(500,input_dim=2,activation='relu',kernel_regularizer=l2 (0.001)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc uracy'])
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=40 00)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test') pyplot.legend()
*****************************************************************************
Practical 6C Repalce L2-L1
*****************************************************************************
from matplotlib import pyplot
from sklearn.datasets import make_moons from keras.models import Sequential from
keras.layers import Dense
from keras.regularizers import l1_l2
X,Y=make_moons(n_samples=100,noise=0.2,random_state=1) n_train=30
trainX,testX=X[:n_train,:],X[n_train:] trainY,testY=Y[:n_train],Y[n_train:]
model=Sequential()
model.add(Dense(500,input_dim=2,activation='relu',kernel_regularizer=l1_l2(l1=0.001,l2=0.001)))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc uracy'])
history=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=40 00)
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test') pyplot.legend()
pyplot.show()
*****************************************************************************
Practical 7 RNN stock
*****************************************************************************
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
# Loading and preprocessing the training dataset
dataset_train = pd.read_csv('/content/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values
# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating data structure for LSTM
X_train = []
Y_train = []
for i in range(60, 1258):
X_train.append(training_set_scaled[i-60:i, 0])
Y_train.append(training_set_scaled[i, 0])
X_train, Y_train = np.array(X_train), np.array(Y_train)
# Reshaping for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# Building the LSTM model
regressor = Sequential()
regressor.add(LSTM(units=50, return_sequences=True,
input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units=1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
# Training the model
regressor.fit(X_train, Y_train, epochs=100, batch_size=32)
# Loading and preprocessing the test dataset
dataset_test = pd.read_csv('/content/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values
# Preparing inputs for prediction
dataset_total = pd.concat((dataset_train['Open'],
dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) -
60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
# Predicting stock prices
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
# Plotting results
plt.plot(real_stock_price, color='red', label='Real Google Stock
Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Stock
Price')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
*****************************************************************************
Practical 8 Encoding and Decodeing
*****************************************************************************
import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
# Define encoding dimension
encoding_dim = 32
# This is our input image
input_img = keras.Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(784, activation='sigmoid')(encoded)
# Create autoencoder model
autoencoder = keras.Model(input_img, decoded)
# Create the encoder model
encoder = keras.Model(input_img, encoded)
# Create the decoder model
encoded_input = keras.Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Load and preprocess dataset
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))
print(X_train.shape)
print(X_test.shape)
# Train autoencoder with training dataset
autoencoder.fit(
X_train, X_train,
epochs=50,
batch_size=256,
shuffle=True,
validation_data=(X_test, X_test)
)
# Predict encoded and decoded images
encoded_imgs = encoder.predict(X_test)
decoded_imgs = decoder.predict(encoded_imgs)
# Plot results
n = 10 # Number of digits to display
plt.figure(figsize=(40, 4))
for i in range(n):
# Display original image
ax = plt.subplot(3, n, i + 1)
plt.imshow(X_test[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Display encoded image
ax = plt.subplot(3, n, i + 1 + n)
plt.imshow(encoded_imgs[i].reshape(8, 4))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# Display reconstruction
ax = plt.subplot(3, n, 2 * n + i + 1)
plt.imshow(decoded_imgs[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
*****************************************************************************
Practical 9 CNN to predict Numbers
*****************************************************************************
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import matplotlib.pyplot as plt
# Download MNIST data and split into train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Plot the first image in the dataset
plt.imshow(X_train[0], cmap='gray')
plt.show()
# Print the shape of the first image
print(X_train[0].shape)
# Reshape data to include the channel dimension
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
# One-hot encode the labels
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
# Print the one-hot encoded labels of the first training sample
print(Y_train[0])
# Define the model
model = Sequential()
# Add model layers
# Learn image features
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,
28, 1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
# Train the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3)
# Predict the classes of the first 4 images in the test set
print(model.predict(X_test[:4]))
# Print the actual results for the first 4 images in the test set
print(Y_test[:4])
*****************************************************************************
Practical 10 Denoise
*****************************************************************************
import keras
from keras.datasets import mnist
from keras import layers
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
# Load and preprocess MNIST data
(X_train, _), (X_test, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))
# Add noise to the data
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0,
scale=1.0, size=X_train.shape)
X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0,
scale=1.0, size=X_test.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_test_noisy = np.clip(X_test_noisy, 0., 1.)
# Display noisy images
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
ax = plt.subplot(1, n, i)
plt.imshow(X_test_noisy[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
# Build the autoencoder model
input_img = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, (3, 3), activation='relu',
padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu',
padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid',
padding='same')(x)
autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Train the autoencoder
autoencoder.fit(
X_train_noisy, X_train,
epochs=3,
batch_size=128,
shuffle=True,
validation_data=(X_test_noisy, X_test),
callbacks=[TensorBoard(log_dir='/tmo/tb', histogram_freq=0,
write_graph=False)]
)
# Predict on the test data
predictions = autoencoder.predict(X_test_noisy)
# Display denoised images
m = 10
plt.figure(figsize=(20, 2))
for i in range(1, m + 1):
ax = plt.subplot(1, m, i)
plt.imshow(predictions[i].reshape(28, 28))
plt.gray()
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
plt.show()
*****************************************************************************
