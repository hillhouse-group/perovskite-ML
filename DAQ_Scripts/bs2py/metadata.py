import json
import uuid
from datetime import datetime
from dateutil.tz import tzlocal

####################################
"""
This contains functions to
generate the metadata for each image
using the CMMCore() object
"""
####################################

class Metadata():
	"""
	For initializing, the MMCore.CMMCore() class object has to be passed
	here. mmc as in the arguement shown is an instance of this class
	"""
	def __init__(self, bs2py):

		time_now = datetime.now(tzlocal())
		self.bs2py = bs2py

		SummaryFields={
			"Slices": 1,
			"UUID": str(uuid.uuid1()),
			"UserName": self.bs2py.mmc.getUserId(),
			"Depth": int(self.bs2py.mmc.getNumberOfComponents()),
			"PixelType": "GRAY16",
			"Time": time_now.strftime('%a %b %d %H:%M:%S PDT %z'),
			"Date": time_now.strftime('%Y-%m-%d'),
			"MetadataVersion": 10,
			"Width": int(self.bs2py.mmc.getImageWidth()),
			"StartTime": time_now.strftime('%Y-%m-%d %H:%M:%S %z'),
			"ChContrastMin": [0],
			"SlicesFirst": False,
			"PixelAspect": 1,
			"MicroManagerVersion": self.bs2py.mmc.getVersionInfo(),
			"ChNames": ["Default0"],
			"IJType": 1,
			"GridRow": 0,
			"Height": int(self.bs2py.mmc.getImageWidth()),
			"NumComponents": int(self.bs2py.mmc.getNumberOfComponents()),
			"GridColumn": 0,
			"Frames": self.bs2py.nrFrames,
			"PixelSize_um": self.bs2py.mmc.getPixelSizeUm(),
			"BitDepth": self.bs2py.mmc.getImageBitDepth(),
			"ComputerName": "HillhouseGroup",
			"Channels": self.bs2py.mmc.getNumberOfCameraChannels(),
			"Source": "Micro-Manager",
			"TimeFirst": True,
			"ChColors": [-1],
			"ChContrastMax": [65535],
			"Positions": 1
		}

		self.init_dict = {
			"Summary" : SummaryFields
		}

		self.modify_dict = {
			"Summary" : SummaryFields
		}

		self.frame_init = {
			self.bs2py.camera + "-TRIGGER SOURCE": "INTERNAL",
			"Core-Focus": self.bs2py.ZAxis,
			"Channel": "Default0",
			"Core-Initialize": "1",
			self.bs2py.XYStage + "-Acceleration X [m/s^2]": "0.2000",
			self.bs2py.camera + "-OUTPUT TRIGGER SOURCE[1]": "READOUT END",
			"FrameIndex": 0,
			self.bs2py.camera + "-PixelType": "16bit",
			self.bs2py.XYStage + "-Backlash Y [um]": "0.0000",
			self.bs2py.camera + "-CONVERSION FACTOR COEFF": "0.4800",
			self.bs2py.ZAxis + "-StepSize [um]": "0.0012",
			self.bs2py.camera + "-OUTPUT TRIGGER KIND[1]": "LOW",
			self.bs2py.camera + "-OUTPUT TRIGGER POLARITY[1]": "NEGATIVE",
			"ROI": "692-692-652-652",
			"DObjective-Trigger": "-",
			self.bs2py.XYStage + "-SpeedY [mm/s]": "10.0000",
			"Camera": self.bs2py.camera,
			self.bs2py.camera + "-OUTPUT TRIGGER SOURCE[2]": "READOUT END",
			self.bs2py.camera + "-TRIGGER ACTIVE": "EDGE",
			"DObjective-Name": "DObjective",
			"Core-ImageProcessor": "",
			"COM4-Parity": "None",
			self.bs2py.camera + "-ReadoutTime": "0.0333",
			self.bs2py.camera + "-Module Version": "16.5.642.5044",
			self.bs2py.camera + "-OUTPUT TRIGGER KIND[0]": "LOW",
			self.bs2py.camera + "-OUTPUT TRIGGER POLARITY[2]": "NEGATIVE",
			self.bs2py.camera + "-Camera Bus": "USB3",
			"Core-AutoShutter": "1",
			self.bs2py.XYStage + "-Name": self.bs2py.XYStage,
			"Coreself.bs2py.XYStage + -": self.bs2py.XYStage,
			self.bs2py.camera + "-OUTPUT TRIGGER DELAY[2]": "0.0000",
			self.bs2py.camera + "-OUTPUT TRIGGER DELAY UNITS": "SECONDS",
			self.bs2py.camera + "-EXPOSURE TIME UNITS": "MILLISECONDS",
			"Electron_Offset": "100.000000",
			"DObjective-State": "0",
			self.bs2py.XYStage + "-StepSizeX [um]": "0.0012",
			"Time": "2020-05-28 22:14:14 -0700",
			self.bs2py.camera + "-TransposeXY": "0",
			self.bs2py.camera + "-CONVERSION FACTOR OFFSET": "100.0000",
			self.bs2py.camera + "-ScanMode": "2",
			"COM4-Name": "COM4",
			"COM4-BaudRate": "57600",
			"ImageNumber": "0",
			self.bs2py.XYStage + "-Port": "COM4",
			self.bs2py.ZAxis + "-Port": "COM4",
			self.bs2py.camera + "-TriggerPolarity": "NEGATIVE",
			"Core-AutoFocus": "",
			self.bs2py.camera + "-MINIMUM ACQUISITION TIMEOUT": "60000",
			self.bs2py.camera + "-CameraID": "S/N: 100243",
			self.bs2py.camera + "-TransposeCorrection": "0",
			"Slice": 0,
			"BitDepth": self.bs2py.mmc.getImageBitDepth(),
			self.bs2py.camera + "-OUTPUT TRIGGER SOURCE[0]": "READOUT END",
			"COM4-Verbose": "1",
			self.bs2py.camera + "-TRIGGER GLOBAL EXPOSURE": "DELAYED",
			"TimeFirst": False,
			self.bs2py.camera + "-OUTPUT TRIGGER POLARITY[0]": "NEGATIVE",
			self.bs2py.camera + "-OUTPUT TRIGGER PERIOD[2]": "0.0010",
			self.bs2py.camera + "-Camera Version": "4.00.A",
			self.bs2py.camera + "-CameraName": "C11440-22CU",
			"ChannelIndex": 0,
			"Core-Shutter": "",
			"DObjective-HubID": "",
			self.bs2py.camera + "-OUTPUT TRIGGER PERIOD[0]": "0.0010",
			"DObjective-Description": "Demo objective turret driver",
			"Core-Camera": self.bs2py.camera,
			"Core-TimeoutMs": "5000",
			"Core-SLM": "",
			self.bs2py.ZAxis + "-Backlash Z [um]": "0.0000",
			self.bs2py.camera + "-OUTPUT TRIGGER PERIOD UNITS": "SECONDS",
			"ElapsedTime-ms": 2766,
			self.bs2py.camera + "-OUTPUT TRIGGER DELAY[0]": "0.0000",
			self.bs2py.XYStage + "-Acceleration Y [m/s^2]": "0.2000",
			"DObjective-Label": "Mitutoyo_5X",
			"Frame": 0,
			"PositionIndex": 0,
			"Width": 652,
			self.bs2py.camera + "-SENSOR MODE": "AREA",
			"Core-ChannelGroup": "",
			self.bs2py.ZAxis + "-SpeedZ [mm/s]": "10.0000",
			self.bs2py.camera + "-OUTPUT TRIGGER PERIOD[1]": "0.0010",
			"PixelSizeUm": 1.448,
			self.bs2py.camera + "-DEFECT CORRECT MODE": "ON",
			self.bs2py.camera + "-Name": self.bs2py.camera,
			"Height": 652,
			self.bs2py.XYStage + "-StepSizeY [um]": "0.0012",
			self.bs2py.XYStage + "-Backlash X [um]": "0.0000",
			self.bs2py.camera + "-Driver Version": "1.2.6.5044",
			self.bs2py.camera + "-OUTPUT TRIGGER DELAY[1]": "0.0000",
			self.bs2py.camera + "-TRIGGER DELAY": "0.0000",
			"COM4-StopBits": "1",
			self.bs2py.camera + "-EXPOSURE FULL RANGE": "DISABLE",
			"COM4-DataBits": "8",
			self.bs2py.ZAxis + "-Name": self.bs2py.ZAxis + "",
			self.bs2py.camera + "-OUTPUT TRIGGER KIND[2]": "LOW",
			self.bs2py.XYStage + "-Description": "Tango XY stage driver adapter",
			self.bs2py.camera + "-Binning": self.bs2py.getCameraBinning(),
			self.bs2py.camera + "HamamatsuImageNr": "0",
			"Binning": self.bs2py.getCameraBinning() ,
			"COM4-Handshaking": "Off",
			self.bs2py.ZAxis + "-Description": "Tango Z axis driver",
			"PixelType": "GRAY16",
			"COM4-DelayBetweenCharsMs": "0.0000",
			"Electron_Coeff": "0.480000",
			"COM4-AnswerTimeout": "500.0000",
			"SlicesFirst": True,
			self.bs2py.camera + "-Exposure": "100.01",
			self.bs2py.XYStage + "-SpeedX [mm/s]": "10.0000",
			"COM4-Description": "Serial port driver (boost:asio)",
			self.bs2py.camera + "-Trigger": "NORMAL",
			self.bs2py.camera + "-Description": "Hamamatsu device adapter 1.8.12.52",
			self.bs2py.camera + "-TransposeMirrorX": "0",
			self.bs2py.camera + "-TransposeMirrorY": "0",
			self.bs2py.XYStage + "-TransposeMirrorY": "0",
			self.bs2py.XYStage + "-TransposeMirrorX": "0",
			self.bs2py.ZAxis + "-Acceleration Z [m/s^2]": "0.2000",
			self.bs2py.camera + "-TRIGGER TIMES": "1",
			"FileName": "MMStack_Pos0.ome.tif",
			"Position": "Default",
			"Core-Galvo": "",
			"CameraChannelIndex": 0,
			"SliceIndex": 0
		}
		# Metadata changes for microscope 2
		if self.bs2py.camera == 'pco_camera':
			self.frame_init[self.bs2py.camera + "-Description"] = "pco generic driver module"
			del self.frame_init[self.bs2py.camera + "HamamatsuImageNr"]

	def inputFrameMD(self, nFrame, ndepth, nChannel, Filename):
		key_name = 'FrameKey-' + '-'.join(map(str, [nFrame, ndepth, nChannel]))
		frame_dict = {
			key_name : self.frame_init
		}

		time_now = datetime.now(tzlocal())

		#Changes to frame_dict:
		frame_dict[key_name][self.bs2py.camera + "-Binning"] = self.bs2py.getCameraBinning() #get Binning
		frame_dict[key_name]["Binning"] = self.bs2py.getCameraBinning() 
		frame_dict[key_name][self.bs2py.camera + "-Exposure"] = self.bs2py.getExposure() #get exposure
		frame_dict[key_name]['Time'] = time_now.strftime('%Y-%m-%d %H:%M:%S %z') #current time
		frame_dict[key_name]['Frame'] = nFrame
		frame_dict[key_name]['Width'] = int(self.bs2py.mmc.getImageWidth())
		frame_dict[key_name]['Height'] = int(self.bs2py.mmc.getImageHeight())
		frame_dict[key_name]['FileName'] = Filename
		
		#Update init_dict
		self.modify_dict.update(frame_dict)

	def writeMetadata(self, path):
		"""Writes the init_dict to the path specified"""
		with open(path, 'w') as file:
			json.dump(self.modify_dict, file)

	def resetMetadict(self):
		self.modify_dict = self.init_dict
