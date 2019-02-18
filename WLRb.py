import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
import sys


def W_FUN(Xi,Tau,x,y):
    W =[]
    
    for i in range (0, len(x)):
        W.append(np.exp(-((x[i,1] - Xi)**2)/(2*Tau*Tau)))
        # print(np.exp(-((x[i,1] - Xi)**2)/(2*Tau*Tau)))
    # print(np.diag(W)) 
    return np.diag(W)


def w_gradient_descent(theta,Tau,X,x,y):
    
    theta_table = np.zeros((len(X),2))
    for i in range(0,len(X)):
        W = W_FUN(X[i],Tau,x,y)
        theta = np.dot((np.linalg.inv(np.dot(np.dot(x.transpose(),W),x))),(np.dot(np.dot(x.transpose(),W),y)))
        theta_table[i,:] = theta.transpose()
    print("theta ="+str(theta))
    return theta_table



def WLR():
    # read the data
    rowsX =[]
    rowsY =[]
    f0 = sys.argv[1]
    f1 = sys.argv[2]
    with open(f0,'r') as fileX:
        read = csv.reader(fileX)
        for row in read:
            rowsX.append([1,float(row[0])])
    # print(rowsX)
    with open(f1,'r') as fileY:
        read = csv.reader(fileY)
        for row in read:
            rowsY.append([float(row[0])])
    #normalize x
    meanX =0.0
    for i in range(0, len(rowsX)):
        meanX += rowsX[i][1]
    meanX = meanX/len(rowsX)
    # print(meanX)
    ##std##
    dev =0.0
    for i in range (0, len(rowsX)):
        dev += (rowsX[i][1]-meanX)**2
    dev = dev/ len(rowsX)
    dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][1] = (rowsX[i][1] -meanX)/dev
    #normalize y
    meanY =0.0
    for i in range(0, len(rowsY)):
        meanY = meanY+rowsY[i][0]
    meanY = meanY/len(rowsY)
    # print(meanY)
    ##std##
    devY =0.0
    for i in range (0, len(rowsY)):
        devY += (rowsY[i][0]-meanY)**2
    devY = devY/ len(rowsY)
    dev = pow(dev,0.5)

    for i in range(0,len(rowsY)):
        rowsY[i][0] = (rowsY[i][0] -meanY)/devY
    

    YY=[]
    for i in range(0,len(rowsX)):
        YY.append(rowsX[i][1])
    X = np.linspace(min(YY),max(YY),100)
    # print(X)


    #matrix formulation
    x = np.asmatrix(rowsX)
    theta = [0.0,0.0]
    
    y = np.asmatrix(rowsY)
    
    # print(len(x))
    tau =float(sys.argv[3])
    theta = w_gradient_descent(theta,tau,X,x,y)
    # print(theta)
    ff = plt.figure()
    x1=[] 
    for i in range(0, len(x)):
        x1.append(x[i,1])
    y1=[]
    y_predict=[] 
    for i in range(0, len(x)):
        y1.append(y[i])
        
    for i in range(0, len(X)):
        y_predict.append(theta[i][0] + (theta[i][1]*X[i]))
        
    plt.scatter(x1,y1)
    plt.plot(X,y_predict,'.r-')
    ff.savefig("tau.png")
    plt.show()
    

if __name__ == '__main__':
    WLR()