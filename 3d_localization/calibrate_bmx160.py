'''
author Manjiang Cao 
e-mail <mcao999@connect.hkust-gz.edu.cn>
this script using for calibrating the bmx160 sensor.
'''
import numpy as np
import matplotlib.pyplot as plt
import serial
import struct

ser = serial.Serial('COM4', 115200)  # Replace 'COM3' with your serial port and baud rate
rawFrame = []

def measure_bmx160_noise():
    rawFrame = []
    DATA_INTERVAL = 0.005
    times = 0
    itration = 500
    accx_list = []
    accy_list = []
    accz_list = []
    RTT_list = []
    while times < itration:
            byte  = ser.read(1)        
            rawFrame += byte
            if rawFrame[-2:]==[13, 10]:
                #print(rawFrame)
                if len(rawFrame) == 17:
                    decimal_data = int.from_bytes(rawFrame[:4],byteorder='big')
                    rssi = bytes(rawFrame[5:9])
                    rssi = rssi.decode('utf-8')
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
                RTT_list.append(decimal_data)
                times  = times + 1
    
    return np.mean(np.array(accx_list)), np.mean(np.array(accy_list)), np.mean(np.array(accz_list)), np.mean(np.array(RTT_list))

x_mean_list = []
y_mean_list = []
z_mean_list = []
RTT_based_mean_list = []
for i in range(10):
    x_noise,y_noise,z_noise,RTT_based = measure_bmx160_noise()
    x_mean_list.append(x_noise)
    y_mean_list.append(y_noise)
    z_mean_list.append(z_noise)
    RTT_based_mean_list.append(RTT_based)

x_mean_noise = np.mean(np.array(x_mean_list))
y_mean_noise = np.mean(np.array(y_mean_list))
z_mean_noise = np.mean(np.array(z_mean_list))
RTT_based_value = np.mean(np.array(RTT_based_mean_list))

print(x_mean_noise)
print(y_mean_noise)
print(z_mean_noise)
print(RTT_based_value)


plt.figure
plt.plot([i for i in range(10)],x_mean_list,c='r',label='x')
plt.plot([i for i in range(10)],y_mean_list,c='b',label='y')
plt.plot([i for i in range(10)],z_mean_list,c='y',label='z')
plt.show()