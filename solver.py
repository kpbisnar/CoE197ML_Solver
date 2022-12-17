import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

#Import CSV file
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

x_train = data_train['x']
y_train = data_train['y']
x_test = data_test['x']
y_test = data_test['y']

def plotter(coords, label=['lab']):
    for points ,l in zip(coords, label):
        x, y = points
        plt.scatter(x, y, label=l)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

#define the function
def function(x,degree,coeffs):
  y = 0
  for deg, coeff in zip(range(degree+1),coeffs):
    y += coeff*(x**deg)
  return y

#loss function
def loss(pred,y):
  MSE = np.square(np.subtract(pred,y)).mean()
  return MSE

#Class Model
class PolynomialModel:
  def __init__(self,degree,coeffs):
    self.degree = degree
    self.coeffs = coeffs

  def forward(self,x):
    y = 0.0
    for i in range (self.degree+1):
      y += self.coeffs[i]*(x**i)
    return y

#Initializing degree and coeffs
degree = 3
coeffs = 2*np.random.rand(degree+1)

#Creating the model and optimizer
model = PolynomialModel(degree,coeffs)
optimizer = optim.SGD(optim.get_parameters(model), lr = 0.01)

epoch = 5
x_tensor = Tensor(x_train.values)
y_tensor = Tensor(y_train.values)
for e in range(epoch):
    for x, y in zip(x_tensor, y_tensor):
        y_pred = model.forward(x)
        l = loss(y_pred,y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"epoch {e}, Loss = {l.data}")

coeffs_pred = model.coeffs
degree_pred = model.degree
print('Coefficients: ', coeffs_pred)
print('Degree: ', degree_pred)


y_prediction = []
for i in range(len(x_test)):
  y_prediction.append(function(x_test[i], degree_pred, coeffs_pred))

#Plotting the predicted function vs the data_test

#Comparing the model on the test data
error = loss(y_prediction,y_test)
plotter([[x_test, y_test],[x_test, y_prediction]], label=['data_test', 'prediction'])

print(f"Test loss: {error}")





