"""
Perceptron Programme without using tensorflow
@Reference : - An Introduction to Machine Learning, Springer, 2021

@author : Ponyathk
"""
#Imports
import numpy as np
import matplotlib as plt
import os
import csv
import math
import random
import pandas as pd

reader = pd.read_csv("F:\PyCharm Community Edition 2022.2.1\Data\rsi.csv")
X = np.array(reader['20MA'])
X = X/np.max(X, axis=0)
y = np.array(reader['Close'])


class Neural_Network(object):
    data = reader()
    weights = []
    def __init__(self):
        #parameters
        self.input_size = 100
        self.hidden_size = 200
        self.output_size = 100
        self.learning_rate = 0.001

        #Weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)

    def sigmoid(s):
            return 1/(1+math.exp(-s))

    def sigmoid_prime(s):
            return s*(1-s)

    def forward(self, X):
        self.z = np.dot(X, self.W2)
        self.z1 = self.sigmoid(self.z)
        self.z2 = np.dot(self.z1,self.W2)
        o = self.sigmoid(self.z2)
        return o

    def backward(self,X,y,o):
        #Error Backpropagation
        self.error = y - o
        self.res_1 = o*(1-y)*(y-o) #Calculating the delta
        self.sum_1 = sum(self.res_1*self.W1)
        x = self.z1

        self.res_2 = x*(1-x)*self.sum_1
        self.delta_1 = self.learning_rate*self.res_1*x
        for a in X:
            self.delta_2 = self.learning_rate*self.res_2*a

        self.W1 += X.T.dot(self.delta_2)
        self.W2 += self.z1.T.dot(self.delta_1)

    def training(self, X, y):
        o = self.forward(X)
        self.backward(X,y,o)

NN = Neural_Network()
loss = np.mean(np.square(y - NN.forward(X)))