import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/yimen/Desktop/HR_comma_sep.csv")

# Overfitting detection
Xtrain,Xvali,ytrain,yvali=train_test_split(X, y, test_size=0.2, random_state=3)

np.random.seed(1)
alpha = 5 # learning rate
beta = np.random.randn(Xtrain.shape[1]) # Random initialization beta
error_rates_train=[]
error_rates_vali=[]
for T in range(200):
    prob = np.array(1. / (1 + np.exp(-np.matmul(Xtrain, beta)))).ravel()
    prob_y = list(zip(prob, ytrain))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(ytrain) 
    error_rate = 0
    for i in range(len(ytrain)):
        if ((prob[i] > 0.5 and ytrain[i] == 0) or (prob[i] <= 0.5 and ytrain[i] == 1)):
            error_rate += 1;
    error_rate /= len(ytrain)
    error_rates_train.append(error_rate)
    
    prob_vali = np.array(1. / (1 + np.exp(-np.matmul(Xvali, beta)))).ravel()  
    prob_y_vali = list(zip(prob_vali, yvali))
    loss_vali = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y_vali]) / len(yvali) 
    error_rate_vali = 0
    for i in range(len(yvali)):
        if ((prob_vali[i] > 0.5 and yvali[i] == 0) or (prob_vali[i] <= 0.5 and yvali[i] == 1)):
            error_rate_vali += 1
    error_rate_vali /= len(yvali)
    error_rates_vali.append(error_rate_vali)
    
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate)+ ' error_vali=' + str(error_rate_vali))

    deriv = np.zeros(Xtrain.shape[1])
    for i in range(len(ytrain)):
        deriv += np.asarray(Xtrain[i,:]).ravel() * (prob[i] - ytrain[i])
    deriv /= len(ytrain)
    
    beta -= alpha * deriv
    
plt.plot(range(50,200), error_rates_train[50:], 'r^', range(50, 200), error_rates_vali[50:], 'bs')
plt.show()


# Gradient checking
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # Random initialize beta

#dF/dbeta0
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  
prob_y = list(zip(prob, y))
loss = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # Loss function
deriv = np.zeros(X.shape[1])
for i in range(len(y)):
    deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
deriv /= len(y)
print('We calculated ' + str(deriv[0]))

delta = 0.0001
beta[0] += delta
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # Demission rate based on current beta
prob_y = list(zip(prob, y))
loss2 = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # Loss function
shouldbe = (loss2 - loss) / delta # (F(b0+delta,b1,...,bn) - F(b0,...bn)) / delta
print('According to definition of gradient, it is ' + str(shouldbe))
