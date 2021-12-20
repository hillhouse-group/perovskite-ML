import MMCorePy
import os
import numpy as np
from datetime import datetime
import tifffile as tf
from skimage.color import rgb2gray
from skimage.util import img_as_uint
from autofocus import autoFocus
import json
import re
from metadata import Metadata
from skimage.io import imread
from skimage.feature import register_translation


class bs2py():
    
    def __init__(self, config_path,
                 exposureMs=25.0,
                 vidTimeMs=5000,
                 nrFrames=50,
                 targetMeanCts=10000): #initial exposure time
        
        self.cfg_path = config_path
        self.mmc = MMCorePy.CMMCore()
        self.mmc.loadSystemConfiguration(config_path)
        self.metadata_file = 'MMStack_Pos0_metadata.txt'

        #Device data
        self.camera = str(self.mmc.getCameraDevice())
        self.ZAxis = str(self.mmc.getFocusDevice())
        self.ZFocusDevice = str(self.mmc.getAutoFocusDevice())
        self.XYStage = str(self.mmc.getXYStageDevice())

        #Acquisition data
        self.mmc.setExposure(exposureMs)
        self.vidTimeMs = vidTimeMs
        self.nrFrames = nrFrames
        self.targetMeanCts = targetMeanCts

        #Metadata object
        self.md = Metadata(self)

    def setZStagePos(self, zPos):
        """sets the zstage to zPos"""
        self.mmc.setPosition(zPos)
        self.mmc.waitForDevice(self.ZAxis)
    
    def setXYStagePos(self, xPos, yPos):
        """sets the XYstage to XPos, yPos (both in microns)"""
        self.mmc.setXYPosition(xPos, yPos)
        self.mmc.waitForDevice(self.XYStage)
        
    def setExposure(self, exposureMs):
        """sets exposure entered in ms"""
        self.mmc.setExposure(exposureMs)
        
    def getExposure(self):
        """returns the camera exposure in ms"""
        return self.mmc.getExposure()

    def getCameraBinning(self):
        """returns the current set binning"""
        
        
        with open(self.cfg_path, 'r') as file:
            args_list = file.readlines()
        
        binning=0
        for i in args_list:
            if 'Binning' in i and self.camera in i:
                binning = i.split(',')[-1]
                if 'x' in binning:
                    binning = int(binning.split('x')[-1])
                else:
                    binning = int(binning)
                break
        
        if binning==0:
            print('No binning specified in '+ self.cfg_path)
        
        return binning


    def setCameraBinning(self, binning, include_x = True):
        """sets the camera binning, include_x = True or False depends on the camera drive
        For HamamatsuHAM_DCAM, include_x must be True
        """
        print('Camera found : ' + self.camera)
        
        with open(self.cfg_path, 'r') as file:
            args_list = file.readlines()
        
        line_found = False
        for i, iline in enumerate(args_list):
            if 'Binning' in iline and self.camera in iline:
                line = iline.split(',')
    			
    			#if binning is like this : 4x4
                if 'x' in line[-1]:
                    if int(line[-1].split('x')[-1]) == binning:
                        print("Binning is already "+str(binning))
                        return
                    line[-1] = str(binning)+'x'+str(binning)+'\n'
    			
    			#if binning is like this : 4
                else:
                    if int(line[-1]) == binning:
                        print("Binning is already "+str(binning))
                        return
                    line[-1] = str(binning)+'\n'
                
                args_list[i] = ','.join(line)
                line_found = True
                break
        
        #If binning is not found in cfg file
        if not line_found:
            req_line = ["ConfigGroup", "System","Startup",self.camera,"Binning", binning]
            if include_x:
                req_line[-1] = str(binning)+'x'+str(binning)+'\n'
            else:
                req_line[-1] = str(binning)+'\n'
            args_list += [','.join(req_line)]
            
        #Write the modified cfg file
        with open(self.cfg_path, 'w') as file:
            args_list = file.writelines(args_list)
        
        #Load the new config file
        self.mmc.loadSystemConfiguration(self.cfg_path)

    
    def setOptExposure(self,starting_guess=25):

        # set initial exposure time if necessary
        # 100 ms is good guess for DF
        # 10 ms is better for bright field
        # PL depends
        self.setExposure(starting_guess)

        """Sets optimal exposure for the camera"""
        maxExpTimeMs = self.vidTimeMs/self.nrFrames
        minExpTimeMs = 1 #Limit of camera

    	# First snap image to set camera exposure time
        self.mmc.snapImage()
        img = self.mmc.getImage()
        
        #Calculating the mean of the pixel intensities
        hist, bin_edges = np.histogram(img, bins=100)
        curMeanPix = np.mean(img)
        
        #Calculating the pixel count at mean intensity and the exposure
        oldExpTime = self.getExposure()
        exposureMs = max(min(oldExpTime * self.targetMeanCts / curMeanPix, maxExpTimeMs), minExpTimeMs)
        
        self.setExposure(exposureMs)

        return exposureMs

    def writeMetadata(self, path, metadata):
        """Writes the metadata file into the vidFolder.
        metadata in the arguements must be the MMCore.Metadata() object
        """
        #gets a metadata dict string
        metastr = metadata.Dump()
        with open(path, 'w') as file:
            json.dump(metastr, file)


    def vidAcquisition(self, vidFolder,
    				   autofocus=False,
                       vidName=None,
                       ref_vid=None,
                       init_x=None,
                       init_y=None,
                       init_z=None,
                       autofocus_kwargs={},
                       exp_guess=25):
        """Acquires videos in burst mode with parameters specified in __init__()
        The videos are saved as ome.tiff ImageJ stacks
        """

        # Extending the path (because the MAX_PATH limit for regular DOS paths is 260)
        vidFolder_list = re.split('[\\\\/]', vidFolder)
        while '' in vidFolder_list or '?' in vidFolder_list:
        	if '' in vidFolder_list:
        		vidFolder_list.remove('')
        	if '?' in vidFolder_list:
        		vidFolder_list.remove('?')

        vidFolder = '\\\\?\\' + '\\'.join(vidFolder_list)

        #clear camera's buffer memory if capacity becomes low
        if self.mmc.getBufferFreeCapacity() < self.nrFrames:
            self.mmc.clearCircularBuffer()


        #If vidname is not provided, use the folder's name
        if vidName==None:
            vidName = re.split('[\\\\/]', vidFolder)[-1] + '.ome.tif'
        
        #Create the directory
        if not os.path.exists(vidFolder):
            os.makedirs(vidFolder)
        
        # Set optimal exposure
        maxExpTimeMs = self.vidTimeMs/self.nrFrames
        exposureMs = self.setOptExposure(starting_guess=exp_guess)

        #breakpoint = os.path.join(vidFolder,'breakpoint')
		#if not os.path.exists(breakpoint):
        #    os.makedirs(breakpoint)

        # Register the stage
        if ref_vid!=None:
        	
        	#if not os.path.exists(vidFolder + '/breakpoint2'):
            #os.makedirs(vidFolder + '/breakpoint2')

            # grab the first frame of the last primary video
        	ref_img = imread(os.path.join(ref_vid,'MMStack_Pos0.ome.tif'))[0,:,:]
        	# snap a new image
	        self.mmc.snapImage()
        	mov_img = self.mmc.getImage()
        	# determine pixel shift
        	pixShift_y, pixShift_x = register_translation(ref_img,mov_img)[0]
        	# calibration factor
        	umPerPix = 38.0/512
        	# shift the stage
        	stageShift_x = umPerPix*pixShift_x
        	stageShift_y = umPerPix*pixShift_y
        	# and move it
        	self.mmc.setXYPosition(init_x + stageShift_x, init_y + stageShift_y)
        	self.mmc.waitForDevice(self.XYStage)

        if autofocus:
            ##Autofocus code -------------------------------------------
            try:
                self.mmc.fullFocus(self.ZFocusDevice)
                zPosPostFocus = self.mmc.getPosition(self.ZFocusDevice)
            except:
                zPosPostFocus = autoFocus(self.mmc, init_z=init_z, **autofocus_kwargs)
            
            #print("After Focusing, zPos = {}".format(zPosPostFocus)) #print the new z position after autofocusing
            #-----------------------------------------------------------
    	
        if not os.path.exists(vidFolder):
            os.mkdir(vidFolder)
        
        #VIDEO ACQUISITION STARTS HERE-------------------------------    
        #get image parameters
        width = int(self.mmc.getImageWidth())
        height = int(self.mmc.getImageHeight())
        depth = int(self.mmc.getNumberOfComponents())
        channels = int(self.mmc.getNumberOfCameraChannels())
        
        # dimensions in TZCYXS order
        img_stack = np.zeros([self.nrFrames, depth, channels, height, width, 1], dtype=np.uint16)
        
        self.mmc.startSequenceAcquisition(self.nrFrames, 0, True)
        frame = 0
        while self.mmc.getRemainingImageCount() > 0 or self.mmc.isSequenceRunning(self.mmc.getCameraDevice()):
            if self.mmc.getRemainingImageCount() > 0:
                frameStartTime = datetime.now()

                #The metadata object
                #md = MMCorePy.Metadata()
                img = self.mmc.popNextImage()
    
                if depth>1:
                    img = rgb2gray(img)
                
                img1 = img_as_uint(img) #16 bit per sample
                img_stack[frame] = np.reshape(img1, [depth, channels, height, width, 1])
                
                itTookPerFrame = datetime.now() - frameStartTime
                self.mmc.sleep(max(maxExpTimeMs - 1000*itTookPerFrame.seconds, 0))

                self.md.inputFrameMD(frame, 0, 0, vidName)
                frame+=1
            
            else:
                self.mmc.sleep(min(0.5 * exposureMs, 20))

        self.mmc.stopSequenceAcquisition() # Finish collecting video
        tf.imwrite(os.path.join(vidFolder, vidName), img_stack, imagej=True)
        #VIDEO ACQUISITION ENDS HERE AND SAVED -----------------------
        
        ### Problem identified after 31 August 2020 update is likely happening in this line:
        self.md.writeMetadata(os.path.join(vidFolder, self.metadata_file))
        self.md.resetMetadict()
    

    def imgAcquisition(self, img_path, autofocus=False,
    					autofocus_kwargs={},exp_guess=25):
        """snaps a single image and saves it as a tif file"""

        # Long path compatibility
        img_path_list = re.split('[\\\\/]', img_path)
        while '' in img_path_list or '?' in img_path_list:
        	if '' in img_path_list:
        		img_path_list.remove('')
        	if '?' in img_path_list:
        		img_path_list.remove('?')

        img_path = '\\\\?\\' + '\\'.join(img_path_list)
        #"""
        # set optimal exposure
        exp_time_ms = self.setOptExposure(starting_guess=exp_guess)
        # roudn exp time to 2 decimal places
        exp_time_ms = np.round(exp_time_ms,2)

        # Autofocus
        if autofocus:
            ##Autofocus code -------------------------------------------
            try:
                self.mmc.fullFocus(self.ZFocusDevice)
                zPosPostFocus = self.mmc.getPosition(self.ZFocusDevice)
            except:
                zPosPostFocus = autoFocus(self.mmc, init_z=None, **autofocus_kwargs)
        #"""
        self.mmc.snapImage()
        img = self.mmc.getImage()
        # append the exposure time to the image name
        img_path = img_path.split('.tif')[0]
        img_path = img_path + '_Exp' + str(exp_time_ms) + 'ms.tif'
        tf.imwrite(img_path, img)
