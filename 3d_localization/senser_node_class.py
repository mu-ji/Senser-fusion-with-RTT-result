'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script implement KF for 2d localization.
'''

import numpy as np
import matplotlib.pyplot as plt
import serial
import struct
import math
import pygame

class senser_node():
    def __init__(self,x,y,z,dt):
        self.position = [x,y,z]
        self.velocity = [0,0,0]
        self.acceleration = [0,0,0]
        self.last_position = [x,y,z]
        self.last_velocity = [0,0,0]
        self.last_acceleration = [0,0,0]
        self.dt = dt

    # for a_z need -9.8
    def updata(self,acc_x,acc_y,acc_z, packet_loss):
        # Save the position, velocity, and acceleration at the previous moment
        self.last_position = self.position
        self.last_velocity = self.velocity
        self.last_acceleration = self.acceleration

        if packet_loss:
            # packet loss, update with the data from the previous moment
            self.acceleration = self.last_acceleration
        else:
            # packet deliver, ipdate acceleration
            self.acceleration = [acc_x,acc_y,acc_z]

        #update velocity
        vx = self.last_velocity[0] + self.acceleration[0]*self.dt
        vy = self.last_velocity[1] + self.acceleration[1]*self.dt
        vz = self.last_velocity[2] + self.acceleration[2]*self.dt
        self.velocity = [vx,vy,vz]

        #updata position
        x = self.last_position[0] + self.velocity[0]*self.dt + 0.5*self.acceleration[0]*self.dt**2
        y = self.last_position[1] + self.velocity[1]*self.dt + 0.5*self.acceleration[1]*self.dt**2
        z = self.last_position[2] + self.velocity[2]*self.dt + 0.5*self.acceleration[2]*self.dt**2
        self.position = [x,y,z]

    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity
    
    def get_acceleration(self):
        return self.acceleration

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
        self.Q = np.array([[10,0,0],
                      [0,10,0],
                      [0,0,10]])
        # measurement matrix
        self.H = np.array([0,
                           0,
                           0,
                           0,
                           0,
                           0])
        # measurement noise matrix
        self.R = np.array([100])

    def update_H(self):
        if self.x_est[0][0]**2+self.x_est[1][0]**2+self.x_est[2][0]**2 == 0:
            self.H = np.array([0,0,0,0,0,0])
        else:
            self.H = np.array([self.x_est[0][0]/(self.x_est[0][0]**2+self.x_est[1][0]**2+self.x_est[2][0]**2)**0.5,
                            self.x_est[1][0]/(self.x_est[0][0]**2+self.x_est[1][0]**2+self.x_est[2][0]**2)**0.5,
                            self.x_est[2][0]/(self.x_est[0][0]**2+self.x_est[1][0]**2+self.x_est[2][0]**2)**0.5,
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

def ML_model_predic(decimal_data,RTT_time_buffer):
    if len(RTT_time_buffer) < 10:
        RTT_time_buffer.append(decimal_data)
    else:
        RTT_time_buffer.pop(0)
        RTT_time_buffer.append(decimal_data)
    
    distance = (float(decimal_data)-20074.659)/(2*16000000)*299792458*0.4
    #distance = (float(np.mean(RTT_time_buffer))-20074.659)/(2*16000000)*222222222
    return distance

#the main function using for visualizing the senser node position
def main():
    ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port and baud rate
    rawFrame = []
    DATA_INTERVAL = 0.0625
    
    KF = Kalman_Filter(DATA_INTERVAL) 

    #init pygame for visualization
    #pygame.init()
    #set windows width and height
    #width = 800
    #height = 600
    #create window and set window title
    #screen = pygame.display.set_mode((width, height))
    #pygame.display.set_caption("My Game")
    
    # 设置坐标轴的起点和终点
    #axis_start = (50, height - 50)
    #axis_end_x = (width - 50, height - 50)
    #axis_end_y = (50, 50)

    # 设置坐标轴的颜色
    #axis_color = (0, 0, 0)
    # set node color and trajectory color
    #node_color = (255, 0, 0)
    #trail_color = (0, 255, 0)

    node_pos = [0, 0]
    trail_points_x = [0]
    trail_points_x_KF = [0]
    trail_points_y = [0]
    bmx160_senser = senser_node(node_pos[0],node_pos[1],0,DATA_INTERVAL)

    dt = DATA_INTERVAL
    
    accx_list = []
    accy_list = []
    accz_list = []
    vx_list = [0]
    ax_list = [0]
    itration = 50
    times = 0
    RTT_time_buffer = []
    while times < itration:

        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        quit()

        byte  = ser.read(1)        
        rawFrame += byte
        if rawFrame[-2:]==[13, 10]:
            #print(rawFrame)
            if len(rawFrame) == 16:
                decimal_data = int.from_bytes(rawFrame[:4],byteorder='big')
                print('RTT time:',decimal_data)
                rssi = bytes(rawFrame[4:8])
                rssi = rssi.decode('utf-8')
                print('rssi:',rssi)
                (x_acc, y_acc, z_acc) = struct.unpack('>hhh', bytes(rawFrame[-8:-2]))                         
                # debug info
                output = 'acc_x={0:<3} acc_y={1:<3} acc_z={2:<3}'.format(
                    x_acc,
                    y_acc,
                    z_acc
                )
                
            rawFrame = []


            acc_reso = 415
            DATA_INTERVAL = 0.0625

            x_acc = float(x_acc)/float(4096)*8
            y_acc = float(y_acc)/float(4096)*8
            z_acc = float(z_acc)/float(4096)*8

            accx_list.append(x_acc)
            accy_list.append(y_acc)
            accz_list.append(z_acc)
            print('acc output:',x_acc,y_acc,z_acc)

            bmx160_senser.updata(x_acc,y_acc,z_acc-9.8,0)
            KF.update_H()
            KF.state_predict(np.array([[x_acc],[y_acc],[z_acc-9.8]]))

            measurement = ML_model_predic(decimal_data,RTT_time_buffer)
            print('measurement:',measurement)
            KF.state_update(measurement)

            KF_position_est = KF.get_x_est()

            trail_points_x_KF.append(KF_position_est[0][0])

            bmx160_position = bmx160_senser.get_position()
            bmx160_v = bmx160_senser.get_velocity()
            bmx160_a = bmx160_senser.get_acceleration()
            print('current position:',bmx160_position)
            node_pos = [bmx160_position[0],bmx160_position[1]]
            trail_points_x.append(bmx160_position[0])
            trail_points_y.append(bmx160_position[1])
            vx_list.append(bmx160_v[0])
            ax_list.append(bmx160_a[0])
            times  = times + 1

    plt.figure()
    ax = plt.subplot(311)
    ax.plot([i for i in range(itration+1)],trail_points_x,c = 'r',label='x position')
    ax.plot([i for i in range(itration+1)],trail_points_x_KF,c = 'b',label='KF x position')
    ax.legend()
    
    ax = plt.subplot(312)
    ax.plot([i for i in range(itration+1)],vx_list,c = 'r',label='x velocity')
    ax.legend()
    ax = plt.subplot(313)
    ax.plot([i for i in range(itration+1)],ax_list,c = 'r',label='x acceleration')
    ax.legend()
    
    plt.show()
    print('accx:',np.mean(np.array(accx_list)))
    print('accy:',np.mean(np.array(accy_list)))
    print('accz:',np.mean(np.array(accz_list)))
    #pygame.quit()

main()