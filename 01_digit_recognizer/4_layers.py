# accuracy  > 95%
 
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

X_train = X_train / 255.0
X_dev = X_dev / 255.0

def init_params():
  W1 = np.random.randn(128, 784) * np.sqrt(1. / 784)
  b1 = np.zeros((128, 1))
  W2 = np.random.randn(64, 128) * np.sqrt(1. / 128)
  b2 = np.zeros((64, 1))
  W3 = np.random.randn(32, 64) * np.sqrt(1. / 64)
  b3 = np.zeros((32, 1))
  W4 = np.random.randn(10, 32) * np.sqrt(1. / 32)
  b4 = np.zeros((10, 1))
  return W1, b1, W2, b2, W3, b3, W4, b4

def ReLU(Z):
  return np.maximum(0, Z)

def softmax(Z):
  expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # stability fix
  return expZ / np.sum(expZ, axis=0, keepdims=True) 
    
def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
  Z1 = W1.dot(X) + b1
  A1 = ReLU(Z1)
  Z2 = W2.dot(A1) + b2
  A2 = ReLU(Z2)
  Z3 = W3.dot(A2) + b3
  A3 = ReLU(Z3)
  Z4 = W4.dot(A3) + b4
  A4 = softmax(Z4)
  return Z1, A1, Z2, A2, Z3, A3, Z4, A4

def one_hot(Y):
  one_hot_Y = np.zeros((10, Y.size))
  one_hot_Y[Y, np.arange(Y.size)] = 1
  return one_hot_Y

def deriv_ReLU(Z):
  return (Z > 0).astype(float)

def back_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W2, W3, W4, X, Y):
  m = Y.size
  one_hot_Y = one_hot(Y)
  dZ4 = A4 - one_hot_Y
  dW4 = (1 / m) * dZ4.dot(A3.T)
  db4 = (1 / m) * np.sum(dZ4, axis=1, keepdims=True)
  dZ3 = W4.T.dot(dZ4) * deriv_ReLU(Z3)
  dW3 = (1 / m) * dZ3.dot(A2.T)
  db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
  dZ2 = W3.T.dot(dZ3) * deriv_ReLU(Z2)
  dW2 = (1 / m) * dZ2.dot(A1.T)
  db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
  dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
  dW1 = (1 / m) * dZ1.dot(X.T)
  db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
  return dW1, db1, dW2, db2, dW3, db3, dW4, db4

def update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha):
  W1 -= alpha * dW1
  b1 -= alpha * db1
  W2 -= alpha * dW2
  b2 -= alpha * db2
  W3 -= alpha * dW3
  b3 -= alpha * db3
  W4 -= alpha * dW4
  b4 -= alpha * db4
  return W1, b1, W2, b2, W3, b3, W4, b4


def get_predictions(A2):
  return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
  print(predictions, Y)
  return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
  W1, b1, W2, b2, W3, b3, W4, b4 = init_params()
  for i in range(iterations):
    Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
    dW1, db1, dW2, db2, dW3, db3, dW4, db4 = back_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W2, W3, W4, X, Y)
    W1, b1, W2, b2, W3, b3, W4, b4 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, alpha)
    if i % 10 == 0:
      acc = get_accuracy(get_predictions(A4), Y)
      print("Iteration", i, "Accuracy:", acc)
  return W1, b1, W2, b2, W3, b3, W4, b4

W1, b1, W2, b2, W3, b3, W4, b4 = gradient_descent(X_train, Y_train, 500, 0.1)

def make_predictions(X, W1, b1, W2, b2, W3, b3, W4, b4):
  _, _, _, _, _, _, _, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
  return get_predictions(A4)

def test_prediction(index, W1, b1, W2, b2, W3, b3, W4, b4):
  current_image = X_train[:, index, None]
  prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3, W4, b4)
  label = Y_train[index]
  print("Prediction:", prediction)
  print("Label:", label)
  plt.gray()
  plt.imshow(current_image.reshape(28, 28) * 255, interpolation='nearest')
  plt.show()

test_prediction(253, W1, b1, W2, b2, W3, b3, W4, b4)

dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3, W4, b4)
print("Dev Accuracy:", get_accuracy(dev_predictions, Y_dev))