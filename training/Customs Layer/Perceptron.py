#Imports
import numpy as np
import math
#Binary Classifier

X=[21,40,60,45] #Attributes
Y=[1,2,2,1] #Class
h=[]

t = [1,2]
#Error backpropagation will not be implemented

weights = np.random.randn(len(X))
learning_rate = 0.001
print(weights)
def sum(weights,X,h, Y):
    prod = np.dot(X,weights)
    print(prod)
    if prod>0:
       np.append(h,t[1])
    elif prod<0:
        np.append(h,t[0])
    print(h)
    if h!=Y:
        for i in range(0,len(Y)):
            error = np.diff(h,Y)
            corr = learning_rate*error*X[i]
            weights += corr
    elif h==Y:
        pass
    return h
sum(weights, X, h, Y)
