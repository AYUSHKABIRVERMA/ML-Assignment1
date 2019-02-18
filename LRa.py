import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def cost_function(theta, x, y):
    # print('theta:  ', theta)
    # print('error: ', np.dot((y-np.dot(x,theta)).transpose(),(y-np.dot(x,theta)))[0,0] /(2*len(x)))
    return np.dot((y-np.dot(x,theta)).transpose(),(y-np.dot(x,theta)))[0,0] /(2*len(x))

def alter_cost_function(theta,x,y):
    new_theta = np.asmatrix(np.zeros((2,1)))
    new_theta[0,0]=theta[0]
    new_theta[1,0]=theta[1]
    theta = new_theta
    # print('theta: ', theta)
    # print('error: ', np.dot((y-np.dot(x,theta)).transpose(),(y-np.dot(x,theta)))[0,0] /(2*len(x)))
    return np.dot((y-np.dot(x,theta)).transpose(),(y-np.dot(x,theta)))[0,0] /(2*len(x))

def gradient(theta,learning_rate,x,y):
    gradient = np.asmatrix(np.zeros((theta.shape[0],1),dtype=float))
    grad0=0.0
    grad1=0.0
    for i in range (0, len(x)):
        grad0 += ((np.dot(x[i,:],theta)-y[i])*x[i,0])[0,0]
        grad1 += ((np.dot(x[i,:],theta)-y[i])*x[i,1])[0,0]         

    gradient[0,0]= (learning_rate/len(x))*grad0
    gradient[1,0]= (learning_rate/len(x))*grad1
    return gradient

def gradient_descent(theta,learning_rate,x,y,ii):
    cost_old = cost_function(theta,x,y)
    tolerance = 1e-3
    iteration =1
    J_t =np.array([])
    while True:
        J_t = np.append(J_t,[theta[0,0],theta[1,0],cost_old])
        grad = gradient(theta,learning_rate,x,y)
        theta = theta-grad
        cost_new = cost_function(theta,x,y)
        if abs(cost_new-cost_old)<tolerance:
            if ii==0:
                print("learning rate = "+str(learning_rate) + " stopping criteria(cost_difference) : "+str(tolerance) +" final parameters= "+str(theta))
            break
        #for debug
        # if iteration%100==0:
        #     print(iteration,theta,cost_new,cost_old)
        cost_old= cost_new
        iteration = iteration+1
    ii=ii+1
    return theta,J_t.reshape(iteration,3),ii



        

def LR():
    # read the data
    rowsX =[]
    rowsY =[]
    ii=0
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
    # dev = pow(dev,0.5)

    for i in range(0,len(rowsX)):
        rowsX[i][1] = (rowsX[i][1] -meanX)/dev
    # #normalize y
    # meanY =0.0
    # for i in range(0, len(rowsY)):
    #     meanY = meanY+rowsY[i][0]
    # meanY = meanY/len(rowsY)
    # # print(meanY)
    # ##std##
    # devY =0.0
    # for i in range (0, len(rowsY)):
    #     devY += (rowsY[i][0]-meanY)**2
    # devY = devY/ len(rowsY)
    # # dev = pow(dev,0.5)

    # for i in range(0,len(rowsY)):
    #     rowsY[i][0] = (rowsY[i][0] -meanY)/devY
    
#    print print('X: ', rowsX)
    # print('Y: ', rowsY)
    #matrix formulation
    x = np.asmatrix(rowsX)
    theta = [[0.0],[0.0]]
    theta = np.asmatrix(theta)
    y = np.asmatrix(rowsY)
    learning_rate =float(sys.argv[3])
    time_gap = float(sys.argv[4])
    # print(learning_rate)
    (theta,J_t,ii) = gradient_descent(theta,learning_rate,x,y,ii)

    (a,b) = np.mgrid[0:2:50j, -1:1:50j]
    m = np.c_[a.flatten(), b.flatten()]
    J = (np.array([alter_cost_function(theta,x,y) for theta in m]).reshape(50,50))
    # print(J)
    plt.ion()
   
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(a, b, J)#, cmap =cm.RdBu_r)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')
    plt.show()
    # print(J_t)
    for jt in J_t:
        ax.plot([jt[0]], [jt[1]], [jt[2]], linestyle='-', color='r', marker='o')#, markersize=2.5)
    plt.pause(time_gap)
    fig.savefig("j_THETA.png")
    plt.show()
    # fig.savefig(plt,"j.png")
    plt.pause(2)
    plt.close()

    #contours
    ff = plt.figure()
    plt.contour(a, b, J, 25, colors="k")
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    for jt in J_t:
            plt.plot([jt[0]], [jt[1]], linestyle='-',color='r', marker='o')#, markersize=2.5)
    plt.pause(time_gap)
    plt.show()
    ff.savefig("CONTOUR.png")
    plt.pause(2)
    plt.close()


    #drawing portion
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
    # dev = pow(dev,0.5)

    for i in range(0,len(rowsY)):
        rowsY[i][0] = (rowsY[i][0] -meanY)/devY
    y = np.asmatrix(rowsY)
    (theta,J_t,ii) = gradient_descent(theta,learning_rate,x,y,ii)
    x1=[] 
    for i in range(0, len(x)):
        x1.append(x[i,1])
    #print(x1)
    
    
    f = plt.figure()
    y1=[]
    y_predict=[]
    for i in range(0, len(x)):
        y1.append(y[i][0,0])
        y_predict.append(theta[0,0] + (theta[1,0]*x[i,1]))
    # print(y1)    
    plt.plot(x1,y1, 'rx')
    plt.plot(x1,y_predict,'.r-')
    
    plt.show()
    f.savefig("Lr.png")
    plt.pause(5)
    #
    # print(rowsX[0][1])
    #normalize y
    # meanY =0.0
    # for i in range(0, len(rowsY)):
        # meanY += rowsY[i]
    # meanY = meanY/len(rowsY)
    ##std##
    # dev =0.0
    # for i in range (0, len(rowsY)):
        # dev += (rowsY[i]-meanY)**2
    # dev = dev/ len(rowsY)
    # dev = pow(dev,0.5)

    # for i in range(0,len(rowsX)):
        # rowsY[i] = (rowsY[i] -meanY)/dev
    #parameters
    # theta =[0.0,0.0]
    # learning_rate =0.0005
    # theta = gradient_descent(theta,learning_rate,rowsX, rowsY)
    # theta2 = [[0],[0]]
    # theta2[0][0] = theta[0]
    # theta2[1][0] = theta[1]
    # # print(np.subtract((np.dot(rowsX,theta2)),rowsY))
    # meanY =0.0
    # for i in range(0, len(rowsY)):
    #     meanY = meanY+rowsY[i][0]
    # meanY = meanY/len(rowsY)
    # # print(meanY)
    # ##std##
    # devY =0.0
    # for i in range (0, len(rowsY)):
    #     devY += (rowsY[i][0]-meanY)**2
    # devY = devY/ len(rowsY)
    # # dev = pow(dev,0.5)

    # for i in range(0,len(rowsY)):
    #     rowsY[i][0] = (rowsY[i][0] -meanY)/devY



    
    # x=[]
    # for i in range(0,len(rowsX)):
    #     x.append(rowsX[i][1])
    # plt.scatter(x,rowsY)
    
    # y = np.dot(theta,rowsX)
    # ax = np.linspace(min(x),max(x),len(x))
    # y=[]
    # for i in range (0, len(rowsY)):
    #     y.append(np.dot(theta,rowsX[i]))
    # plt.plot(ax,y)
    

    # plt.show()




if __name__ == '__main__':
    LR()
