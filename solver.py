import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from tinygrad.tensor import Tensor
import tinygrad.nn.optim as optim

#Load the CSV file
data_train = pd.read_csv('data_train.csv')
data_test = pd.read_csv('data_test.csv')

x_train = data_train['x']
y_train = data_train['y']
x_test = data_test['x']
y_test = data_test['y']

#Converting  to Tensor
x_trn = Tensor(x_train.values) 
y_trn = Tensor(y_train.values)
x_tst = Tensor(x_test.values) 
y_tst = Tensor(y_test.values)

#Function to plot datas (from https://github.com/izzajalandoni/Deep-Learning-Helper/blob/main/Machine_Learning/Optimization/GradientDescent.ipynb)
def plotter(coords, label=['lab']):
    for points ,l in zip(coords, label):
        x, y = points
        plt.scatter(x, y, label=l)
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

plotter([[x_test, y_test],[x_train, y_train]], label=['data_test', 'data_train'])

def function(x,degree,coeffs):
  y = 0
  for deg, coeff in zip(range(degree+1),coeffs):
    y += coeff*(x**deg)
  return y

#Polynomial Model
class Polynomial:
  def __init__(self, degree):
    self.degree = degree
    self.coeffs = Tensor(np.random.randn(degree + 1))
    
  def forward(self,x):
    y = 0.0
    for i in range (self.degree+1):
      y += self.coeffs[i]*(x**i)
    return y

#define the MSE 
def MSE(y_pred,y):
  MSE = ((y_pred.sub(y)).square()).mean()
  return MSE

#Initialize the degree and learning rate
degree = 2
lr = 0.01

#Create the model and initialize the optimizer
model = Polynomial(degree)
optimizer = optim.SGD(optim.get_parameters(model), lr = lr)

#Training the model
epoch = 100
for e in range(epoch):
  for x, y in zip(x_train, y_train):
    y_pred = model.forward(x)

    l = MSE(y_pred,y)

    l.backward()
    optimizer.step()

    optimizer.zero_grad()

  print(f"epoch {e+1}, Loss = {l.data}")

coeffs_pred = (model.coeffs).numpy()
degree_pred = model.degree
print('Coefficients: ', coeffs_pred)
print('Degree: ', degree_pred)


y_prediction = []
for i in range(len(x_test)):
  y_prediction.append(function(x_test[i], degree_pred, coeffs_pred))

#Plotting the predicted function vs the data_test
#Comparing the model on the test data
error = MSE(y_prediction,y_test)
plotter([[x_test, y_test],[x_test, y_prediction]], label=['data_test', 'prediction'])

print(f"Test loss: {error}")




