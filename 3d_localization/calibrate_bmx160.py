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
DATA_INTERVAL = 0.0625
times = 0
itration = 100

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