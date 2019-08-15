import numpy as np
import matplotlib.pyplot as plt

X = np.array([1,2,3,4,5])
Y = np.array([5,7,9,11,13])


def gradient_descent(x,y):
    m = 0
    b = 0
    learning_rate = 0.08
    steps = 6500
    N = len(x)
    for i in range(1,steps+1):
        y_p = m*x + b
        cost = (1/N)*sum([val**2 for val in (y-y_p)])
        Dm = -(2/N) * sum(x*(y-y_p))
        Db = -(2/N) * sum(y-y_p)
        m = m - learning_rate * Dm
        b = b - learning_rate * Db        
        print("m={}, b={}, cost={}, step={}".format(m,b,cost,i))
    return m,b,cost

param = gradient_descent(X,Y)
m = param[0]
b = param[1]
error = param[2]

Yp = X*m + b

plt.scatter(X,Y,color='r',marker='+')
plt.plot(X,Yp,color='g')

plt.show()






