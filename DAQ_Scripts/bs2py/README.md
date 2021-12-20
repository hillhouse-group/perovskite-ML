# bs2py
PLVA codes - Converting beanshell codes to python to integrate with Keithley data collection ***(Currently, works only on single points)***

# Files to download for testing on Lab's PC:
1. bs2py.py
2. autofocus.py
3. Photoconductivity2.cfg
4. lumencor.py
5. PL_LD_Tr_2020_06_24.py
6. plaq.py

# How to use:
Instead of keeping micromanager window running in the background, we can use these python codes to get primary videos and images.
The repo consists of three main files:
1. bs2py.py
2. autofocus.py
3. Photoconductivity2.cfg

The 3rd file is same as the one present in C:/Program Files/Micromanager1.4.17, but I slightly modified it to get 4x4 binning. Let's use the one present here in the github repo to run the python code and not disturb the one in the software's folder.

The class in **bs2py.py** is to be callable, while the functions in autofocus.py are utility functions to be used in the bs2py.py. An example of how to use the function can be seen in PLVA_single_point.py which does the same job as Stoddard_PLVAaq_061019_lockin_multipleGradients.bsh program when run on micromanager for a single point acquisistion.


## Instructions to test on local PC:

### PLVA_single_point.py
This file has a sample code how to use the **bs2py class** from bs2py.py file.

ONE-TIME SETUP:
1. Install Micromanager1.4 using MSI and python2.7
2. **During python 2.7 installation, uncheck the registry option to not make python2.7 default**
3. Also uncheck the last option in the list.
4. Add C:/ProgramFiles/Micromanager1.4/ to PYTHONPATH
5. Create a copy of python.exe and rename it to python27.exe. Make sure both files are present within Python27 folder.
6. Change Scripts/pip.exe to Scripts/pip27.exe
7. Add both python27 and python27/Scripts folders to PATH variable.
8. pip27 install numpy scipy matplotlib ipython jupyter pandas sympy nose
9. pip27 install scikit-image
10. pip27 install tifffile faulthandler
11. change the CONFIG_FILE while running PLVA_single_point.py

## To test on Lab's PC:
From steps **1-10** already done on lab's PC, only step **11** has to be done.
1. change CONFIG_FILE to ***Photoconductivity2.cfg***
2. Open git bash in the same folder:
<pre>python27 PLVA_single_point.py</pre>

## Issues:
Exposure not adjusting.



# More Information:

## hyperspectral TIFF generation:
A typical TIFF file would have 1 - 4 samples per pixel (see [this](http://www.mit.edu/afs.new/sipb/project/scanner/arch/i386_rhel3/bin/html/vuesc13.htm))

## MMCorepy.cpp References:

* [Source Code](https://github.com/micro-manager/micro-manager/blob/master/MMCore/MMCore.cpp)
* [Functions Documentation](https://valelab4.ucsf.edu/~MM/doc/MMCore/html/class_c_m_m_core.html)
* [Tifffile package Source code](https://github.com/cgohlke/tifffile/blob/master/tifffile/tifffile.py) : No proper documentation available
* [Image metadata object](https://github.com/micro-manager/micro-manager/blob/master/MMDevice/ImageMetadata.h) as used in MMCore.cpp
* [OughtaFocus Source Code](https://github.com/micro-manager/micro-manager/blob/master/autofocus/src/main/java/org/micromanager/autofocus/OughtaFocus.java)

