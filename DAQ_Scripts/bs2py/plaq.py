#!/usr/bin/env python27
#Must run this file with python 2.7

##############################################################################
# This code must be run parallelly along with the keithley's code.
# This is just like beanshell code running parallelly, but the difference is,
# we can start this new parallel process from the keithley's code using multiprocess.
# Otherwise, the camera loads each time we call this file separately and that makes it
# very slow.

# Must call this code like this
# subprocess.Popen(['python27', path_to_this_file, config_file_path, save_path, number_of_cycles])
##############################################################################
import bs2py as bs
import numpy as np
import argparse
from time import sleep
import os
import re
from datetime import datetime
from scipy.interpolate import interp1d



#take arguements from the terminal
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('args', nargs=argparse.REMAINDER)
args_func = parser.parse_args()
args = args_func.args

# The only arguements allowed are 
# args[0] : The config_file_path
# args[1] : The save_path_folder
# args[2] : Number of cycles/acquisitions
# args[3] : ngrads = # of gradients
# args[4] : ngrad_points = # of points on each gradient
# args[5] : Path to np file with initial start, middle and end cordinates
# args[6] : Measurement start cycle
# args[7] : if 0, then no dark field; if 1, then take dark field images
# args[8] : if 0, then no bright field; if 1, then take bright field images
# args[9] : if str, then use this basename for video folder names and filenames; if 0. then use the args[1] as basename
# args[10]: initial guess for z_PL
# args[11]: initial guess for z_DF

if not os.path.exists(args[1]):
	os.makedirs(args[1])

#If gradients are present
ngrads = int(args[3])
ngrad_points = int(args[4])

#Whether dark field is needed or not
try:
	DF = int(args[7])
except:
	DF = 0

#same for bright field
try:
	BF = int(args[8])
except:
	BF = 0
    
# Initial guess for PL z-height
try:
	zpos_PL = int(args[10])
except:
	zpos_PL = 0

# Initial guess for DF z-height
try:
	zpos_DF = int(args[11])
except:
	zpos_DF = 0

# initial guess for PL video exposure time (ms)
PL_exp = 25


def generate_points(filepath):
    """
    The file must contain a np. array with ngrads*3 rows and 3 columns.
    The 3 rows for each gradient correspond to start, middle and end composition.
    The 3 columns correspond to x, y, z.
    
    Interpolates the anchors to get the intermediate cordinates
    """
    anchors = np.load(filepath)
    points = np.zeros((ngrads*ngrad_points, 3))
    for i in range(ngrads):
        start = anchors[3*i + 0, :]  # j = 0
        middle = anchors[3*i + 1, :] # j= int(0.5*(ngrad_points-1))
        end = anchors[3*i + 2, :]    # j = (ngrad_points-1)
        
        if ngrad_points==1:
            points[i, :] = start

        else:
        	# if anchors correspond to cordinates of each ngrad_point
        	if ngrads*ngrad_points == anchors.shape[0]:
        		for j in range(ngrad_points):
        			ind = ngrad_points*i + j
        			points[ind, :] = anchors[ngrad_points*i + j, :]

        	# Otherwise, use the anchors as the start, middle and end coordinates
        	else:
	            for j in range(ngrad_points):
	                ind = ngrad_points*i + j
	                if j<=int(0.5*(ngrad_points-1)):
	                    points[ind, :] = start + j*(middle - start)/(int(0.5*(ngrad_points-1)))
	                else:
	                    points[ind, :] = middle + (j - int(0.5*(ngrad_points-1)))*(end-middle)/(ngrad_points - 1 - int(0.5*(ngrad_points-1)))
    
    return points

def point_loc(grad_index, pos_number, points):
    """Returns the point cordinates index from gradient index and position index"""
    ind = ngrad_points*grad_index + pos_number
    return ind

#initialize the class with the config file
mm = bs.bs2py(args[0], exposureMs=25.0)

if not args[9] == '0':
	baseName = args[9]
else:
	baseName = re.split('[\\\\/]', args[1])[-1]
check_time = 2 #seconds

if not (ngrads ==1 and ngrad_points == 1):
    points = generate_points(args[5])
    autofocus_kwargs = {
        'zstep':100, #mu
        'SearchRange_um':500,
        'fine_search_range':200,
        'fine_step':20,
        }
else:
    autofocus_kwargs = {
        'zstep':5, #mu
        'SearchRange_um':50,
        'fine_search_range':0,
        'fine_step':1,
        }

count=int(args[6])
while count < int(args[2]):
    
    grad_index, pos_number = divmod(count, ngrad_points) # 0s for spin coated samples
    time_count, grad_index = divmod(grad_index, ngrads)
    
    #Set the video folder's name
    vidName = baseName + "_grad" + str(grad_index) + "_loc" + str(pos_number) + "_time" + str(time_count)
    vidFolder = os.path.join(args[1], vidName)

    ##### 8/18/2020 addition for video registration
    # Get the previous video's name for registration
    if time_count > 0:
        prev_vid = baseName + "_grad" + str(grad_index) + "_loc" + str(pos_number) + "_time" + str(time_count-1)
        prev_vidFolder = os.path.join(args[1], prev_vid)

    #Checking for the temporary directory
    while not os.path.exists(os.path.join(args[1], 'temp')):
        sleep(check_time)
    
    #Move the XY stage and the z stage to appropriate positions
    if not (ngrads ==1 and ngrad_points == 1):
        ind = point_loc(grad_index, pos_number, points)
        x,y,z = points[ind, :]
        mm.setZStagePos(z)
        mm.setXYStagePos(x, y)
    else:
    	x = None
    	y = None
        prev_vidFolder = None
    
    # if PL height given, move the stage there
    if zpos_PL != 0:
        mm.setZStagePos(zpos_PL)
    
    #Video Acquisition
    vidStartTime = datetime.now()

    #mm.vidAcquisition(vidFolder, autofocus=True, vidName="MMStack_Pos0.ome.tif") #The acquisition
    if time_count == 0:
        mm.vidAcquisition(vidFolder, autofocus=True, vidName="MMStack_Pos0.ome.tif", ref_vid=None, init_x=x, init_y=y, autofocus_kwargs=autofocus_kwargs,exp_guess=PL_exp) #The acquisition
    else:
        mm.vidAcquisition(vidFolder, autofocus=True, vidName="MMStack_Pos0.ome.tif", ref_vid=prev_vidFolder, init_x=x, init_y=y, autofocus_kwargs=autofocus_kwargs,exp_guess=PL_exp) #The acquisition
    time_per_acquisition = datetime.now() - vidStartTime

    # update the exposure time
    PL_exp = mm.getExposure()

    print("Time {} - Grad {} - Loc {} : Exposure Time (ms): {}, took : {}s".format(time_count, grad_index, pos_number, PL_exp, time_per_acquisition.seconds)) #Printing the exposure time
    
    #Updating the z position based on previous value
    if not (ngrads ==1 and ngrad_points == 1):
        points[ind,2] = mm.mmc.getPosition(mm.ZAxis)

    #If dark field is also wanted
    if DF == 1:
        DF_path = vidFolder + '_DF.tif'
        os.makedirs(os.path.join(args[1], 'DF_temp'))
        if zpos_DF != 0:
            mm.setZStagePos(zpos_DF)
        while os.path.exists(os.path.join(args[1], 'DF_temp')):
            sleep(int(check_time/2))
        DF_autofocus_kwargs = {
            'zstep':10, #mu
            'SearchRange_um':500,
            'fine_search_range':0,
            'fine_step':1,
        }
        mm.imgAcquisition(DF_path, autofocus=True, autofocus_kwargs=DF_autofocus_kwargs,exp_guess=100)
        os.makedirs(os.path.join(args[1], 'DF_temp'))

        #If bright field is also wanted
    if BF == 1:
        BF_path = vidFolder + '_BF.tif'
        os.makedirs(os.path.join(args[1], 'BF_temp'))
        while os.path.exists(os.path.join(args[1], 'BF_temp')):
            sleep(int(check_time/2))
        BF_autofocus_kwargs = {
            'zstep':20, #mu
            'SearchRange_um':500,
            'fine_search_range':20,
            'fine_step':1,
        }
        mm.imgAcquisition(BF_path, autofocus=True, autofocus_kwargs=BF_autofocus_kwargs,exp_guess=10)
        os.makedirs(os.path.join(args[1], 'BF_temp'))

    #Removing the temporary directory
    os.rmdir(os.path.join(args[1], 'temp'))
    count+=1
        
    