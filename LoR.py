import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
import sys


def sigmoid(x,y,theta):
    h=[]
    for i in range(0,len(x)):
        h.append(1/(1 + np.exp(-np.dot(x[i,:],theta)[0,0])))
    return h


def log_likelihood(x,y,theta):
    sum =0.0
    for i in range(0, len(x)):
        sum = sum+(y[i]*np.log(sigmoid(x,y,theta)[i])+ (1-y[i])*np.log(1-sigmoid(x,y,theta)[i]))
    return sum

def D_h(theta,x,y):
    h = []
    for i in range(0, len(x)):
        h.append((1/(1 + np.exp(-np.dot(x[i,:],theta)[0,0])))*(1-(1/(1 + np.exp(-np.dot(x[i,:],theta)[0,0])))))
    return np.diag(h)


def calculate_hessian(x,y,theta):
    D = D_h(theta,x,y)
    return np.dot(np.dot(x.transpose(),D),x)

def calculate_gradient(x,y,theta):
    h=sigmoid(x,y,theta)
    h0 = np.asmatrix(np.zeros((y.shape[0],1),dtype=float))
    for i in range(0,len(y)):
        h0[i,0]=h[i]
    # print(np.dot(x.transpose(),(h0-y)))
    return np.dot(x.transpose(),(h0-y))

    

def newton(x,y,theta):
    old_log = log_likelihood(x,y,theta)
    tolerance = 1e-7
    iteration =0
    hashish = calculate_hessian(x,y,theta)
    print("Hessian init ="+str(hashish))
    while True:
        grad = calculate_gradient(x,y,theta)
        hashish = calculate_hessian(x,y,theta)
        H_inv = np.linalg.inv(hashish)
        grad_mat = np.dot(H_inv,grad)
        theta_new = theta - grad_mat
        new_log = log_likelihood(x,y,theta_new)
        if np.linalg.norm(grad_mat)<tolerance:
            print("THETA = "+str(theta_new)+"ITERATION ="+str(iteration)+ " Stopping condition on grad diff = " +str(tolerance)+ " Hassian fin =" +str(hashish))
            break
        theta = theta_new
        iteration = iteration+1
    return theta




def LoR():
    rowsX =[]
    rowsY =[]
    f0 = sys.argv[1]
    f1 = sys.argv[2]
    with open(f0,'r') as fileX:
        read = csv.reader(fileX)
        for row in read:
            rowsX.append([1,float(row[0]),float(row[1])])
    # print(rowsX)
    with open(f1,'r') as fileY:
        read = csv.reader(fileY)
        for row in read:
            rowsY.append([float(row[0])])
    #normalize
    meanX0 =0.0
    for i in range(0, len(rowsX)):
        meanX0 += rowsX[i][2]
    meanX0 = meanX0/len(rowsX)
    # print(meanX0)
    ##std##
    dev =0.0
    for i in range (0, len(rowsX)):
        dev += (rowsX[i][2]-meanX0)**2
    dev = dev/ len(rowsX)
    # dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][2] = (rowsX[i][2] -meanX0)/dev

    meanX1 =0.0
    for i in range(0, len(rowsX)):
        meanX1 += rowsX[i][1]
    meanX1 = meanX1/len(rowsX)
    # print(meanX1)
    ##std##
    dev =0.0
    for i in range (0, len(rowsX)):
        dev += (rowsX[i][1]-meanX1)**2
    dev = dev/ len(rowsX)
    # dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][1] = (rowsX[i][1] -meanX1)/dev
    #normalize y
    x = np.asmatrix(rowsX)
    theta = [[0.0],[0.0],[0.0]]
    theta = np.asmatrix(theta)
    y = np.asmatrix(rowsY)
    theta = newton(x,y,theta)
    x2_predicted=[]
    x1_datapoint=[]
    x2_datapoint=[]
    # print(theta[0,0])
    # print(theta[1,0])
    # print(theta[2,0])
    for i in range(0, len(x)):
        x2_predicted.append((-theta[0,0]-theta[1,0]*x[i,1])/theta[2,0]) 
        x1_datapoint.append(x[i,1])
        x2_datapoint.append(x[i,2])
    # print(x2_datapoint)
    # print(x1_datapoint)
    # print(x2_predicted)
    ff = plt.figure()
    plt.plot(x1_datapoint,x2_predicted)
    plt.scatter(x1_datapoint,x2_datapoint)
    ff.savefig("logistic.png")
    plt.show()


if __name__ == '__main__':
    LoR()