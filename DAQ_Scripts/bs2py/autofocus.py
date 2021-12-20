from skimage.filters import sobel
import numpy as np
from skimage.util import img_as_ubyte, img_as_uint, img_as_float

def getStagePos(mmc):
    """returns the stage positions the stage positions. mmc is the MMCorePy.CMMCore() object.
    The cnfiguration file should have been loaded already"""

    xy = mmc.getXYStageDevice()
    zStage = mmc.getFocusDevice()

    xPos = mmc.getXPosition(xy)
    yPos = mmc.getYPosition(xy)
    zPos = mmc.getPosition(zStage)

    return (xPos, yPos, zPos)


def setStagePos(mmc, zPos):
    """sets the zstage to zPos"""
    zStage = mmc.getFocusDevice()
    mmc.setPosition(zPos)
    mmc.waitForDevice(zStage)    


def calculateEdgeScore(img, CropFactor=1):
	"""
	https://github.com/micro-manager/micro-manager/blob/master/autofocus/src/main/java/org/micromanager/autofocus/OughtaFocus.java
	This is the default score used by gui.getAutofocus().fullFocus() in micromanager, rewritten in python
	"""
	h, w = img.shape
	new_w = int(w*CropFactor)
	new_h = int(h*CropFactor)
	new_x = int((w-new_w)/2)
	new_y = int((h-new_h)/2)
    
	img = img[new_y: new_y+new_h-1, new_x: new_x+new_w-1]
	img_edges = sobel(img)
	edgeMean = np.mean(img_edges)
	imgMean = np.mean(img)
    
	return edgeMean/imgMean

def getImage(mmc):
    """returns the snapped image"""
    mmc.snapImage()
    img = mmc.getImage()
    img1 = img_as_float(img) #16 bit per sample
        
    return img1


def autoFocus(mmc,
			zstep=100, #mu
			SearchRange_um=500,
			ftol=0.001,
			CropFactor=0.8,
			init_z=None,
            scoreCropFactor=0.8,
            fine_search_range=200,
            fine_step=20,
            offset=0 # um, to account for Edges algorithm usually underestimating optimal point
			):
	"""	sets the z pos to that with maximum EdgeScore and returns z pos.
	OugthaFocus code used here."""

	if CropFactor<1:
		x, y, w, h = mmc.getROI()
		new_w = int(w*CropFactor)
		new_h = int(h*CropFactor)
		new_x = int(x + (w-new_w)/2)
		new_y = int(y + (h-new_h)/2)
		mmc.setROI(new_x, new_y, new_w, new_h)
		mmc.waitForDevice(mmc.getCameraDevice()) 

	if init_z == None:
		_ , _ , init_z = getStagePos(mmc)
        
    # Define the search range
	test_zPos = np.arange(init_z-SearchRange_um/2, init_z+SearchRange_um/2, zstep)
	score_arr = test_zPos.copy()
	init_score = calculateEdgeScore(getImage(mmc), CropFactor=scoreCropFactor)
    
    # Try different positions and get scores
	for num, i in enumerate(test_zPos):
		setStagePos(mmc, i)
		score_arr[num] = calculateEdgeScore(getImage(mmc), CropFactor=scoreCropFactor)
		#print("Edge score : " + str(score_arr[num]))
	
    #Get the position with the maximum score
	opt_zPos = test_zPos[np.argmax(score_arr)] + offset
	max_zscore = np.amax(score_arr)
	
	# Fine search
	if fine_search_range>0:
		test_zPos = np.arange(opt_zPos-fine_search_range/2, opt_zPos+fine_search_range/2, fine_step)
		score_arr = test_zPos.copy()
		
		for num, i in enumerate(test_zPos):
			setStagePos(mmc, i)
			score_arr[num] = calculateEdgeScore(getImage(mmc), CropFactor=scoreCropFactor)
			#print("Edge score : " + str(score_arr[num]))
		
		opt_zPos = test_zPos[np.argmax(score_arr)] + offset
		max_zscore = np.amax(score_arr)

    # Change the position only if the new position results in a score higher than the previous one
    # by value greater than the tolerance value
	if np.abs(init_score - max_zscore)>ftol:
		setStagePos(mmc, opt_zPos)
		z_to_return = opt_zPos
	else:
		setStagePos(mmc, init_z)
		z_to_return = init_z

	if CropFactor<1:
		mmc.setROI(x, y, w, h)
		mmc.waitForDevice(mmc.getCameraDevice()) 

	return z_to_return





