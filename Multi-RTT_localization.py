'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script realize the visualization of multi RTT receiver 
'''

import numpy as np
import matplotlib.pyplot as plt
import serial
import struct
import math
import pygame

class Kalman_Filter():
    def __init__(self,dt):
        #x.T = [Px,Py,Pz,Vx,Vy,Vz]
        self.x_pred = np.array([[0],[0],[0],[0],[0],[0]])
        self.x_est = np.array([[0],[0],[0],[0],[0],[0]])
        self.p_pred = np.eye(6)
        self.p_est = np.eye(6)
        self.y = 0      #residual
        self.K = np.array([[0],[0],[0],[0],[0],[0]])
        self.A = np.array([[1,0,0,dt,0,0],
                      [0,1,0,0,dt,0],
                      [0,0,1,0,0,dt],
                      [0,0,0,1,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])
        #u.T = [ax,ay,az]
        self.B = np.array([[0.5*dt**2,0,0],
                      [0,0.5*dt**2,0],
                      [0,0,0.5*dt**2],
                      [dt,0,0],
                      [0,dt,0],
                      [0,0,dt]])
        # process noise matrix
        self.Q = np.array([[0.1,0,0],
                      [0,0.1,0],
                      [0,0,0.1]])
        # measurement matrix
        self.H = np.array([1,
                           0,
                           0,
                           0,
                           0,
                           0])
        # measurement noise matrix
        self.R = np.array([0.01])

    def update_H(self):
        if self.x_est[0][0]**2+self.x_est[1][0]**2+self.x_est[2][0]**2 == 0:
            self.H = np.array([1,0,0,0,0,0])
        else:
            self.H = np.array([1,
                            0,
                            0,
                            0,
                            0,
                            0])
    def state_predict(self,u):
        self.x_pred = self.A.dot(self.x_est) + self.B.dot(u)
        self.p_pred = self.A.dot(self.p_est).dot(self.A.T)

    def state_update(self,measurement):
        self.y = measurement - self.H.dot(self.x_pred)
        #self.K = self.p_pred.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.p_pred).dot(self.H.T)+self.R))
        self.K = self.p_pred.dot(self.H.T)/(self.H.dot(self.p_pred).dot(self.H.T)+self.R)
        self.x_est = self.x_pred + self.K.reshape((6,1)).dot(self.y.reshape((1,1)))
        self.p_est = (np.eye(6)-self.K.dot(self.H)).dot(self.p_pred)

    def get_x_pred(self):
        return self.x_pred

    def get_x_est(self):
        return self.x_est

    def show_parameters(self):
        print(self.x_pred)
        print(self.p_pred)
        print(self.x_est)
        print(self.p_est)
        print(self.K)
        return

def RTT_range_predic(decimal_data,RTT_time_buffer):
    if len(RTT_time_buffer) < 200:
        RTT_time_buffer.append(decimal_data)
    else:
        RTT_time_buffer.pop(0)
        RTT_time_buffer.append(decimal_data)

    #distance = (float(decimal_data)-20074.659)/(2*16000000)*299792458*0.4
    distance = (float(np.mean(RTT_time_buffer))-20074.659)/(2*16000000)*222222222
    return distance

def estimate_point_position(x1, y1, x2, y2, x3, y3, r1, r2, r3):
    grid_size = 0.1
    max_x = max(x1+r1,x2+r2,x3+r3)
    min_x = min(x1-r1,x2-r2,x3-r3)
    max_y = max(y1+r1,y2+r2,y3+r3)
    min_y = min(y1-r1,y2-r2,y3-r3)
    x_range = np.arange(min_x,max_x,grid_size)
    y_range = np.arange(min_y,max_y,grid_size)
    inner_list = []
    est_x = 0
    est_y = 0
    for i in x_range:
        for j in y_range:
            if (i-x1)**2+(j-y1)**2 < r1**2 and (i-x2)**2+(j-y2)**2 < r2**2 and (i-x3)**2+(j-y3)**2 < r3**2:
                inner_list.append([i,j])
    
    for i in inner_list:
        est_x = est_x + i[0]*(1/len(inner_list))
        est_y = est_y + i[1]*(1/len(inner_list))

    return est_x,est_y

#the main function using for visualizing the senser node position
def main():
    ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port and baud rate
    rawFrame = []
    DATA_INTERVAL = 0.005
    
    KF = Kalman_Filter(DATA_INTERVAL) 

    dt = DATA_INTERVAL

    receiver1_measurement_list = []
    receiver2_measurement_list = []
    receiver3_measurement_list = []
    itration = 1000
    times = 0
    receiver1_buffer = []
    receiver2_buffer = []
    receiver3_buffer = []

    receiver1_measurement = 0
    receiver2_measurement = 0
    receiver3_measurement = 0

    receiver_one_position = [0, 0]
    receiver_two_position = [5, 0]
    receiver_three_positon = [0, 5]
    recevier_id = 0

    est_x_list = []
    est_y_list = []

    while times < itration:
    #while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            #print(rawFrame)
            if len(rawFrame) == 17:
                receiver_id = rawFrame[0]
                print('receiver_id:',recevier_id)
                decimal_data = int.from_bytes(rawFrame[1:5],byteorder='big')
                print('RTT time:',decimal_data)
                rssi = bytes(rawFrame[5:9])
                rssi = rssi.decode('utf-8')
                print('rssi:',rssi)
                (x_acc, y_acc, z_acc) = struct.unpack('>hhh', bytes(rawFrame[-8:-2]))                         
                # debug info
                output = 'acc_x={0:<3} acc_y={1:<3} acc_z={2:<3}'.format(
                    x_acc,
                    y_acc,
                    z_acc
                )
            
            if receiver_id == 1:
                receiver1_measurement = RTT_range_predic(decimal_data, receiver1_buffer)
                receiver1_measurement_list.append(receiver1_measurement)
            if receiver_id == 2:
                receiver2_measurement = RTT_range_predic(decimal_data, receiver2_buffer)
                receiver2_measurement_list.append(receiver2_measurement)
            if receiver_id == 3:
                receiver3_measurement = RTT_range_predic(decimal_data, receiver3_buffer)
                receiver3_measurement_list.append(receiver3_measurement)

            rawFrame = []
            acc_reso = 415
            DATA_INTERVAL = 0.005

            x_acc = float(x_acc)/float(4096)*8-0.15
            y_acc = float(y_acc)/float(4096)*8+0.19
            z_acc = float(z_acc)/float(4096)*8+2

            print('acc output:',x_acc,y_acc,z_acc)

            est_x, est_y = estimate_point_position(0,0,5,0,0,5,receiver1_measurement,receiver2_measurement,receiver3_measurement)
            est_x_list.append(est_x)
            est_y_list.append(est_y)

            times  = times + 1
            print('times:',times)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(est_x_list,est_y_list,c='b',label='estimate prosition')
    ax.scatter(est_x_list[-1],est_y_list[-1],c='r')
    ax.scatter(0,0,c='b')
    ax.scatter(5,0,c='b')
    ax.scatter(0,5,c='b')
    ax.legend()
    ax.grid()

    plt.show()
    #pygame.quit()

main()