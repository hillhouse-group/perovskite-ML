#!/usr/bin/env python27
#Must run this file with python 2.7

#ONE-TIME SETUP:
#Step0: Install Micromanager1.4 using MSI and python2.7
#Step 0.1: During python 2.7 installation, uncheck the registry option to make python2.7 default
#Step 0.2: Also uncheck the last option in the list.
#Step1: Add C:/ProgramFiles/Micromanager1.4/ to PYTHONPATH
#Step2: Copy python.exe to python27.exe (both must be present) within Python27 folder
#Step3: Change Scripts/pip.exe to Scripts/pip27.exe
#Step4: Add both python27 and python27/Scripts folders to PATH variable.
#Step4: pip27 install numpy scipy matplotlib ipython jupyter pandas sympy nose
#Step5: pip27 install scikit-image
#Step6: pip27 install tifffile faulthandler
#Step8: change the CONFIG_FILE
#Step9: PYTHONFAULTHANDLER=1 in bash terminal

import os
from datetime import datetime
from time import sleep
import re
import sys

sys.path.append("../microscope_2")
sys.path.append("../")
import bs2py as bs
import faulthandler

faulthandler.enable()

#------------------------------------------------------------------------------
#VARIABLES TO CHANGE
#------------------------------------------------------------------------------
#path to directory to save data
rootDirName = "C:/Users/Jac/Desktop/bs2py/others/test123"
nRepeats = 3 # number of times to repeat video acquisition
standby_time = 2 #s Time between two acquisitions
CONFIG_FILE = '../microscope_2/InitialConfiguration.cfg'

#------------------------------------------------------------------------------
#MICROMANAGER CODE
#------------------------------------------------------------------------------
baseName = re.split('[\\\\/]', rootDirName)[-1] #the last folder name in rootDirName
vidNum = 1
mm = bs.bs2py(CONFIG_FILE)

#For a single spin coated sample
#i,j only used in vidName below
j=0 #Because a single gradient
i=0 #Because a single composition in the single gradient

while (vidNum <= nRepeats):
    vidName = baseName + "_grad" + str(j) + "_loc" + str(i) + "_time" + str(vidNum-1)
    vidFolder = os.path.join(rootDirName, vidName)
    vidStartTime = datetime.now()
    

    # Acquires primary vids and saves them
    print("\nSaving video in "+ vidFolder)
    mm.vidAcquisition(vidFolder, autofocus=True)
    time_per_acquisition = datetime.now() - vidStartTime
    print("{} : Exposure Time (ms): {}, time : {}s".format(vidNum, mm.getExposure(), time_per_acquisition.seconds)) #Printing the exposure time
    
    
    vidNum+=1

    #time between start of videos (in ms) - i.e., frame period in secondary video
    sleep(standby_time)









