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

class KalmanFilter:
    def __init__(self, dt, noise_cov):
        self.dt = dt  # 时间步长
        self.A = np.array([[1, 0], [0, 1]])  # 状态转移矩阵
        self.H = np.array([[1, 0]])  # 观测矩阵
        self.Q = noise_cov * np.eye(2)  # 系统噪声协方差矩阵
        self.R = noise_cov  # 观测噪声方差

        self.x = np.zeros((2, 1))  # 状态估计向量
        self.P = np.zeros((2, 2))  # 状态估计协方差矩阵

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(2) - np.dot(K, self.H)), self.P)

#the main function using for visualizing the senser node position
def main():
    ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port and baud rate
    rawFrame = []
    DATA_INTERVAL = 0.0625
    
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
    trail_points_y = [0]
    bmx160_senser = senser_node(node_pos[0],node_pos[1],0,DATA_INTERVAL)

    dt = DATA_INTERVAL
    noise_cov = 0.01
    kf = KalmanFilter(dt, noise_cov)
    
    accx_list = []
    accy_list = []
    accz_list = []
    vx_list = [0]
    ax_list = [0]
    itration = 50
    times = 0
    while times < itration:

        #for event in pygame.event.get():
        #    if event.type == pygame.QUIT:
        #        pygame.quit()
        #        quit()

        byte  = ser.read(1)        
        rawFrame += byte

        if rawFrame[-2:]==[13, 10]:
            if len(rawFrame) == 8:        
                #print(rawFrame)                    
                (x_acc, y_acc, z_acc) = struct.unpack('>hhh', bytes(rawFrame[:-2]))                         
                # debug info
                output = 'acc_x={0:<3} acc_y={1:<3} acc_z={2:<3}'.format(
                    x_acc,
                    y_acc,
                    z_acc
                )
                
            rawFrame = []


            acc_reso = 415
            DATA_INTERVAL = 0.0625

            x_acc = float(x_acc)/float(4096)
            y_acc = float(y_acc)/float(4096)
            z_acc = float(z_acc)/float(4096)

            accx_list.append(x_acc)
            accy_list.append(y_acc)
            accz_list.append(z_acc)
            print('acc output:',x_acc,y_acc,z_acc)

            bmx160_senser.updata(x_acc,y_acc,z_acc-9.8,0)

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

            # draw node positon and trajectory in window
            #screen.fill((255, 255, 255))  # set background color
            #pygame.draw.circle(screen, node_color, node_pos, 5)  # 绘制点
            #pygame.draw.lines(screen, trail_color, False, trail_points)  # 绘制轨迹

            # 绘制X轴
            #pygame.draw.line(screen, axis_color, axis_start, axis_end_x, 2)

            # 绘制Y轴
            #pygame.draw.line(screen, axis_color, axis_start, axis_end_y, 2)

            # 更新窗口显示
            #pygame.display.flip()
    plt.figure()
    ax = plt.subplot(311)
    ax.plot([i for i in range(itration+1)],trail_points_x,c = 'r',label='x position')
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