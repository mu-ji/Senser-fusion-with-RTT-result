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

    RTT_list = []
    while times < itration:
            byte  = ser.read(1)        
            rawFrame += byte
            if rawFrame[-2:]==[13, 10]:
                #print(rawFrame)
                if len(rawFrame) == 10:
                    decimal_data = int.from_bytes(rawFrame[:4],byteorder='big')
                    rssi = bytes(rawFrame[4:8])
                    rssi = rssi.decode('utf-8')
                    
                    
                rawFrame = []


                acc_reso = 415
                DATA_INTERVAL = 0.0625


                RTT_list.append(decimal_data)
                times  = times + 1
    
    return np.mean(np.array(RTT_list))


RTT_based_mean_list = []
for i in range(10):
    RTT_based = measure_bmx160_noise()

    RTT_based_mean_list.append(RTT_based)


RTT_based_value = np.mean(np.array(RTT_based_mean_list))

print(RTT_based_value)

