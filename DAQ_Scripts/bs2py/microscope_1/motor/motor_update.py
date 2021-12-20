import serial
import math
import random
import time
#Change this port based on the USB port used for connecting Arduino
#The port's name can be checked in the Device Manager

class Stepper_motor:
    
    def __init__(self):
        self.port = 'COM7'
        self.b_rate = 9600
        self.t_out = 1
        self.ser = serial.Serial(self.port, self.b_rate, timeout = self.t_out)
        self.mode = 3
        self.angle = -180
    
    def update_pos_angle(self, add_angle):
        self.angle = self.angle + add_angle
        self.angle = ((self.angle/360) - math.floor(self.angle/360))
        if self.angle<=0.5 :
            self.angle = int(self.angle*360)
        else:
            self.angle = int((self.angle-1)*360)
    
    def run(self, mode):
        if (type(mode) == int):
            #Mode = 1 :
            #Mode = 2 :
            #Mode = 3 :
            #Mode = 4 : 
            #Mode = 5 : Intermediate
            angle = 0
            if mode == 1:
                angle = -self.angle
            elif mode == 2:
                if abs(-self.angle - 90)<abs(-self.angle + 270):
                    angle = -self.angle - 90
                else:
                    angle = -self.angle + 270
            elif mode == 3:
                if abs(-self.angle - 180)<abs(-self.angle + 180):
                    angle = -self.angle - 180
                else:
                    angle = -self.angle + 180
            elif mode == 4:
                if abs(-self.angle + 90)<abs(-self.angle - 270):
                    angle = -self.angle + 90
                else:
                    angle = -self.angle - 270
            elif mode == 5:
                angle = self.angle + 45
                
            self.ser.write(b'%a' % angle)
            self.mode = mode
            self.update_pos_angle(angle)            

    def run_angle(self, angle):
        if (type(angle) == int):
            self.ser.write(b'%a' % angle)
            self.update_pos_angle(angle)
            
    def reset(self):
        self.ser.close()
        self.ser.open()
        self.mode = 3
        self.angle = -180
    
    def close(self):
        self.ser.close()


"""
motor = Stepper_motor()

print("Initial mode : {}".format(motor.mode))

motor.mode = 3
motor.angle = 180
motor.run(motor.mode)
print("Initial mode : {}".format(motor.mode))
time.sleep(2)

#12 secs for rotation between cubes at mils = 7
for i in range(1000):
    motor.run(4)
    print('Iter : {} - mode '.format(i+1) + '{}'.format(motor.mode) + '  expecting...')
    time.sleep(6)
    motor.run(3)
    print('Iter : {} - mode '.format(i+1) + '{}'.format(motor.mode) + '  expecting...')
    time.sleep(6)

"""
"""
#motor.close()
"""
"""
motor.action(mode) for selecting mode
motor.reset() for resetting the motor's connection
motor.close() for closing the connection with motor

To change motor's port, change motor.port = <new string value>
"""