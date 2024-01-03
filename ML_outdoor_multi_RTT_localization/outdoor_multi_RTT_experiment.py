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

from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim

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

class RegressionNet(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RegressionNet, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.elu = nn.ELU()
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.elu = nn.ELU()
            self.fc3 = nn.Linear(hidden_size, output_size)



        def forward(self, x): 
            out = self.fc1(x)
            out = self.elu(out)
            out = self.fc2(out)
            out = self.elu(out)
            out = self.fc3(out)
            return out
        
def RTT_range_predic(decimal_data,RTT_time_buffer,based_RTT_time,rssi,receiver_RSSI_buffer):

    if len(RTT_time_buffer) < 500:
        RTT_time_buffer.append(decimal_data - based_RTT_time)
    else:
        RTT_time_buffer.pop(0)
        RTT_time_buffer.append(decimal_data - based_RTT_time)
    
    if len(receiver_RSSI_buffer) < 500:
        receiver_RSSI_buffer.append(rssi)
    else:
        receiver_RSSI_buffer.pop(0)
        receiver_RSSI_buffer.append(rssi)

    distance = (float(np.mean(RTT_time_buffer)))/(2*16000000)*299792458*0.4
    #distance = (float(np.mean(RTT_time_buffer))-20074.659)/(2*16000000)*222222222
    return distance

def ML_range_predic(RTT_time_buffer,rssi_buffer,RTT_based_time,model):
    model_input = compute_model_input(RTT_time_buffer, rssi_buffer)
    with torch.no_grad():
        outdoor_predictions = model(model_input)
    return outdoor_predictions

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

def draw_circle(x1,y1,r):
    theta = np.linspace(0, 2*np.pi, 100)  # 角度从0到2π均匀分布
    x = x1 + r * np.cos(theta)  # x坐标
    y = y1 + r * np.sin(theta)  # y坐标
    return x,y


def GMM_filter(data):
    #best_aic,best_bic = compute_number_of_components(data,1,5)
    #n_components = best_aic  # 设置成分数量
    n_components = 2
    gmm = GaussianMixture(n_components=n_components)
    try:
        gmm.fit(data)
    except:
        lenghts = len(data)
        gmm.fit(data.reshape((lenghts,1)))
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_

    return means,covariances,weights

def compute_model_input(RTT_time_list, RSSI_list):
    RTT_mean = np.mean(np.array(RTT_time_list))
    RTT_var = np.var(np.array(RTT_time_list))
    RSSI_mean = np.array(np.mean(RSSI_list))
    RSSI_var = np.var(np.mean(RSSI_list))

    means,covariances,weights = GMM_filter(np.array(RTT_time_list))

    RTT_component1_mean = means[0][0]
    RTT_component1_var = covariances[0][0][0]
    RTT_component2_mean = means[1][0]
    RTT_component2_var = covariances[1][0][0]
    RTT_component1_weight = weights[0]
    RTT_component2_weight = weights[1]

    model_input = [RTT_component1_mean,RTT_component1_var,
                   RTT_component2_mean,RTT_component2_var,
                   RTT_component1_weight,RTT_component2_weight,
                   RTT_mean,RTT_var,
                   RSSI_mean,RSSI_var]

    model_input = torch.from_numpy(np.array(model_input)).float()
    return model_input 



#the main function using for visualizing the senser node position
def main():
    ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port and baud rate
    rawFrame = []
    DATA_INTERVAL = 0.005
    based_RTT_time = [20074.806,20074.263,20073.28]
    
    KF = Kalman_Filter(DATA_INTERVAL)
    input_size = 10
    hidden_size = 512
    output_size = 1
    outdoor_model = RegressionNet(input_size, hidden_size, output_size)
    #outdoor_model.load_state_dict(torch.load('ML_outdoor_multi_RTT_localization/experiment3_outdoor_model'))

    outdoor_model.load_state_dict(torch.load('ML_outdoor_multi_RTT_localization/indoor_model'))

    dt = DATA_INTERVAL

    receiver1_measurement_list = []
    receiver2_measurement_list = []
    receiver3_measurement_list = []

    itration = 2000
    times = 0

    receiver1_RTT_buffer = []
    receiver2_RTT_buffer = []
    receiver3_RTT_buffer = []
    receiver1_RSSI_buffer = []
    receiver2_RSSI_buffer = []
    receiver3_RSSI_buffer = []

    receiver1_measurement = 0
    receiver2_measurement = 0
    receiver3_measurement = 0

    receiver_one_position = [0, 0]
    receiver_two_position = [5.25, 0]
    receiver_three_positon = [5.25, 2.5]
    sender_true_position = [1.4, 1.4]
    receiver_id = 0

    est_x_list = []
    est_y_list = []

    while times < itration:
    #while True:
        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 17:
                receiver_id = rawFrame[0]
                #print('receiver_id:',receiver_id)
                decimal_data = int.from_bytes(rawFrame[1:5],byteorder='big')
                #print('RTT time:',decimal_data)
                rssi = bytes(rawFrame[5:9])
                rssi = int(rssi.decode('utf-8'))
                #print('rssi:',rssi)
                (x_acc, y_acc, z_acc) = struct.unpack('>hhh', bytes(rawFrame[-8:-2]))                         
                # debug info
                output = 'acc_x={0:<3} acc_y={1:<3} acc_z={2:<3}'.format(
                    x_acc,
                    y_acc,
                    z_acc
                )
            
            if receiver_id == 1:
                receiver1_measurement = RTT_range_predic(decimal_data, receiver1_RTT_buffer, based_RTT_time[0],rssi,receiver1_RSSI_buffer)
                #receiver1_measurement = ML_range_predic(decimal_data, rssi,
                #                                        receiver1_RTT_buffer, receiver1_RSSI_buffer,
                #                                        based_RTT_time[0], outdoor_model)
                receiver1_measurement_list.append(receiver1_measurement)
            if receiver_id == 2:
                receiver2_measurement = RTT_range_predic(decimal_data, receiver2_RTT_buffer, based_RTT_time[1],rssi,receiver2_RSSI_buffer)
                #receiver2_measurement = ML_range_predic(decimal_data, rssi,
                #                                        receiver2_RTT_buffer, receiver2_RSSI_buffer,
                #                                        based_RTT_time[1], outdoor_model)
                receiver2_measurement_list.append(receiver2_measurement)
            if receiver_id == 3:
                receiver3_measurement = RTT_range_predic(decimal_data, receiver3_RTT_buffer, based_RTT_time[2],rssi,receiver3_RSSI_buffer)
                #receiver2_measurement = ML_range_predic(decimal_data, rssi,
                #                                        receiver3_RTT_buffer, receiver3_RSSI_buffer,
                #                                        based_RTT_time[2], outdoor_model)
                receiver3_measurement_list.append(receiver3_measurement)

            rawFrame = []
            acc_reso = 415
            DATA_INTERVAL = 0.005

            x_acc = float(x_acc)/float(4096)*8-0.15
            y_acc = float(y_acc)/float(4096)*8+0.19
            z_acc = float(z_acc)/float(4096)*8+2

            #print('acc output:',x_acc,y_acc,z_acc)
            #print('RTT measurement:', receiver1_measurement,receiver2_measurement,receiver3_measurement)

            est_x, est_y = estimate_point_position(receiver_one_position[0],receiver_one_position[1],
                                                   receiver_two_position[0],receiver_two_position[1],
                                                   receiver_three_positon[0],receiver_three_positon[1],
                                                   receiver1_measurement,receiver2_measurement,receiver3_measurement)
            #print(est_x,est_y)
            est_x_list.append(est_x)
            est_y_list.append(est_y)

            times  = times + 1
            print('times:',times)
    
    ML_receiver1_measurement = ML_range_predic(receiver1_RTT_buffer, receiver1_RSSI_buffer,based_RTT_time[0], outdoor_model)
    ML_receiver2_measurement = ML_range_predic(receiver2_RTT_buffer, receiver2_RSSI_buffer,based_RTT_time[1], outdoor_model)
    ML_receiver3_measurement = ML_range_predic(receiver3_RTT_buffer, receiver3_RSSI_buffer,based_RTT_time[2], outdoor_model)
    print(ML_receiver1_measurement,ML_receiver2_measurement,ML_receiver3_measurement)
    ML_est_x, ML_est_y = estimate_point_position(receiver_one_position[0],receiver_one_position[1],
                                                   receiver_two_position[0],receiver_two_position[1],
                                                   receiver_three_positon[0],receiver_three_positon[1],
                                                   ML_receiver1_measurement,ML_receiver2_measurement,ML_receiver3_measurement)
    
    print('1:',np.mean(receiver1_RTT_buffer))
    print('2:',np.mean(receiver2_RTT_buffer))
    print('3:',np.mean(receiver3_RTT_buffer))

    # compute 1 meters and 2 meters error boundary
    _1meter_err_x,_1meter_err_y = draw_circle(sender_true_position[0],sender_true_position[1],1)
    _2meter_err_x,_2meter_err_y = draw_circle(sender_true_position[0],sender_true_position[1],2)


    ML_err = ((ML_est_x - sender_true_position[0])**2 + (ML_est_y - sender_true_position[1])**2)**0.5
    RTT_err = ((est_x_list[-1] - sender_true_position[0])**2 + (est_y_list[-1] - sender_true_position[1])**2)**0.5
    plt.figure()
    ax = plt.subplot(211)
    #ax.plot(est_x_list,est_y_list,c='b',label='estimate prosition')
    ax.scatter(est_x_list[-1],est_y_list[-1],c='r',label='RTT estimate position')
    ax.scatter(ML_est_x,ML_est_y,c='y',label='ML estimate position')
    ax.scatter(receiver_one_position[0],receiver_one_position[1],c='b')
    ax.scatter(receiver_two_position[0],receiver_two_position[1],c='b')
    ax.scatter(receiver_three_positon[0],receiver_three_positon[1],c='b')
    ax.scatter(sender_true_position[0],sender_true_position[1],c='g',label='true position')
    ax.plot(_1meter_err_x, _1meter_err_y, c = 'g', linestyle = '--', label = '1 meters error boundary')
    ax.plot(_2meter_err_x, _2meter_err_y, c = 'g', linestyle = '--', label = '2 meters error boundary')
    ax.legend()
    ax.set_title('ML_err:{} RTT_err:{}'.format(ML_err,RTT_err))
    ax.grid()
    
    print(est_x_list[-1],est_y_list[-1])
    print(ML_est_x,ML_est_y)

    ax = plt.subplot(212)
    ax.plot([i for i in range(len(receiver1_measurement_list))],receiver1_measurement_list,c='r',label='receiver 1')
    ax.plot([i for i in range(len(receiver2_measurement_list))],receiver2_measurement_list,c='g',label='receiver 2')
    ax.plot([i for i in range(len(receiver3_measurement_list))],receiver3_measurement_list,c='b',label='receiver 3')
    ax.legend()
    ax.grid()


    plt.show()
    #pygame.quit()
main()