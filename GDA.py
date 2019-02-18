import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
import sys

def phi(x,y):
    ret =0.0
    for i in range(0, len(y)):
        if y[i]==1.0:
            ret =ret+1
    return ret/(len(y))

def u1(x,y):
    den =0
    a=[0.0,0.0]
    a = np.asmatrix(a)
    for i in range(0, len(y)):
        if y[i]==1.0:
            den = den+1
            a = np.add(x[i,:],a)
    return (1/den)*a

def u0(x,y):
    den =0
    a=[0.0,0.0]
    a = np.asmatrix(a)
    for i in range(0, len(y)):
        if y[i]==0.0:
            den = den+1
            a = np.add(x[i,:],a)
    return (1/den)*a

def sigma(x,y,u_0,u_1):
    sigmaa = np.asmatrix(np.zeros((2,2),dtype=float))
    for i in range(0, len(y)):
        if y[i]==1:
            sigmaa = np.add(sigmaa,np.dot((x[i,:].transpose())-(u_1.transpose()), (x[i,:]-u_1) ))
        else:
            sigmaa = np.add(sigmaa,np.dot((x[i,:].transpose())-(u_0.transpose()), (x[i,:]-u_0) ))
    return (1/len(y))*sigmaa

def prediction(phi,u0,u1,sigma):
    theta =[0.0,0.0]
    theta = np.asmatrix(theta)
    theta = np.dot((u0-u1),np.linalg.inv(sigma))
    theta0 = (1/2)*(np.dot(np.dot(u0,np.linalg.inv(sigma)),u0.transpose())) - (1/2)*(np.dot(np.dot(u1,np.linalg.inv(sigma)),u1.transpose())) - np.log((1-phi)/phi)
    return (theta0,theta) 
    
def sigma_notequal(x,y,u_0,u_1):
    sigma0 = np.asmatrix(np.zeros((2,2),dtype=float))
    sigma1 = np.asmatrix(np.zeros((2,2),dtype=float))
    den0=0;
    den1=0;
    for i in range(0, len(y)):
        if y[i]==1:
            den1 =den1+1
            sigma1 = np.add(sigma1,np.dot((x[i,:].transpose())-(u_1.transpose()), (x[i,:]-u_1) ))
        else:
            den0 =den0+1
            sigma0 = np.add(sigma0,np.dot((x[i,:].transpose())-(u_0.transpose()), (x[i,:]-u_0) ))
    return ((1/den0)*sigma0,(1/den1)*sigma1) 
def bound(A,B,C,p):
    return p @ A @ p.transpose() +B@p+C

def prediction_notequal(phi,u0,u1,sigma0,sigma1):
    termA = np.subtract( np.linalg.inv(sigma0) ,np.linalg.inv(sigma1))
    termB = -2* (np.subtract(np.dot(u0,np.linalg.inv(sigma0)), np.dot(u1,np.linalg.inv(sigma1))))
    termC =  (np.dot(np.dot(u0,np.linalg.inv(sigma0)),u0.transpose())) - (np.dot(np.dot(u1,np.linalg.inv(sigma1)),u1.transpose())) +2* np.log((1-phi)/phi) *np.linalg.det(sigma0)/np.linalg.det(sigma1)
    a,b = np.mgrid[-1:1:50j, -2:2:50j]
    Z = np.c_[a.flatten(),b.flatten()]
    # print(np.shape(termC))
    quad_boundary = np.array([bound(termA,termB,termC,a) for a in Z])
    quad = quad_boundary.reshape(50,50)
    return (a,b,quad)


def GDA():
    rowsX =[]
    rowsY =[]
    f0 = sys.argv[1]
    f1 = sys.argv[2]
    with open(f0,'r') as fileX:
        read = fileX.readlines()
        for row in read:
            rowsX.append([float(row.strip().split()[0]),float(row.strip().split()[1])])
    # print(rowsX)
    with open(f1,'r') as fileY:
        read = fileY.readlines()
        for row in read:
            if row.strip().split()[0]=="Canada":
                rowsY.append([float(1)])
            else:
                rowsY.append([float(0)])
                

    meanX0 =0.0
    for i in range(0, len(rowsX)):
        meanX0 += rowsX[i][0]
    meanX0 = meanX0/len(rowsX)
    # print(meanX0)
    ##std##
    dev =0.0
    for i in range (0, len(rowsX)):
        dev += (rowsX[i][0]-meanX0)**2
    dev = dev/ len(rowsX)
    dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][0] = (rowsX[i][0] -meanX0)/dev

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
    dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][1] = (rowsX[i][1] -meanX1)/dev
    
    x = np.asmatrix(rowsX)
    y = np.asmatrix(rowsY)
    # print(phi(x,y))
    iii = sys.argv[3] 
    print("WHEN  Σ0 = Σ1 = Σ")
    print(phi(x,y))
    print("u0 ="+str(u0(x,y)))
    print("u0 ="+str(u1(x,y)))
    print("sigma ="+ str(sigma(x,y,u0(x,y),u1(x,y))))




    (theta0,thata1) = prediction(phi(x,y),u0(x,y),u1(x,y),sigma(x,y,u0(x,y),u1(x,y)))
    (sigma0,sigma1) = sigma_notequal(x,y,u0(x,y),u1(x,y))
    print("WHEN  Σ0 != Σ1 ")
    print(phi(x,y))
    print("u0 ="+str(u0(x,y)))
    print("u0 ="+str(u1(x,y)))
    print("sigma0 ="+ str(sigma0))
    print("sigma1 ="+ str(sigma1))
    (p,q,quad) = prediction_notequal(phi(x,y),u0(x,y),u1(x,y),sigma0,sigma1)
    # print(theta0,thata1)
    x1_datapoint=[]
    x2_datapoint=[]
    x2_predicted=[]
    for i in range(0, len(x)):
        x2_predicted.append((-theta0[0,0]-thata1[0,0]*x[i,0])/thata1[0,1])
        x1_datapoint.append(x[i,0])
        x2_datapoint.append(x[i,1])
    ff = plt.figure()
    color = [ 'b' if i else 'g' for i in y ]
    plt.scatter(x1_datapoint,x2_datapoint, c = color)
    plt.plot(x1_datapoint,x2_predicted)
    plt.contour(p,q,quad,[0],color ="red")
    ff.savefig("GDA.png")
    plt.show()

if __name__ == '__main__':
    GDA()