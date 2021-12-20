import serial
ser = serial.Serial('COM6', 9600,timeout=1)
#Change this port based on the USB port used for connecting Arduino
#The port's name can be checked in the Device Manager

def motor(action):
    if (type(action) == int):
        #Mode = 1 :
        #Mode = 2 :
        #Mode = 3 :
        #Mode = 4 : 
        #Mode = -1 : Intermediate
        #Mode = 0 : No action
        ser.write(b'%a' % action)
        
    elif (action == 'reset'):
        ser.close()
        ser.open()
    
    elif (action == 'close'):
        ser.close()

# motor(mode_number) for selecting mode
# motor('reset') for resetting the motor's connection
# motor('close') for closing the connection with motor