# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:17:33 2019

@author: Ryan Stoddard
"""

#import packages
from __future__ import print_function, division, absolute_import
import numpy as np
import pandas as pd
import scipy as sp
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import subprocess
import json

from matplotlib.animation import FFMpegWriter
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import os
from scipy.stats import skew
from scipy.stats import kurtosis
import platform
from skimage import io
from tqdm import tqdm
import seaborn as sns
from natsort import natsorted

import sys
sys.path.append('../../')
sys.path.append('../')

import SQ_calcs
import utils
from stats import Constants, calibrate_videos, calc_plqy_qfls
from stats import generate_2d_round_window, generate_radii_angle_array
from utils import (load_primary_videos, nsun_generator, load_exposure)
from stats import find_plqy_75
import Ld_analysis_batch
const = Constants()
kB = const.kb
q = const.q


if platform.system() == 'Linux':
    ffmpeg_path = '/usr/bin/ffmpeg'
elif platform.system() == 'Windows':
    ffmpeg_path = 'C:/Program Files/ffmpeg/bin/ffmpeg.exe'
elif platform.system() == 'Darwin':  # MacOS
    raise NotImplementedError('Not sure where the FFMEPG installed on mac')
else:
    raise NotImplementedError('OS {} currently not supported'.format(platform.system()))
plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path


#change default plot settings
mpl.rcParams['figure.figsize'] = [9.6, 7.2]
font = {'family' : 'DejaVu Sans',
        'weight' : 'bold',
        'size'   : 24}

mpl.rc('font', **font)
mpl.rc('axes', linewidth=3)

import warnings
warnings.filterwarnings("ignore")

#%%
#Constants
pi = math.pi


# Provide a list of directories containing the raw experimental data for the degradation
# experiments you wish to analyze:
    
directory_list = ['Trial_Data/Film_Data/PL_PC_T_DF_1sun_25C_40RH_air_150nm',
                  ]



# Example for how to enter a long path
# r'\\?\C:\Users\hughadm\Documents\Users\Ryan\hpdb\data\plva\200812_Gradients\Cs30MA35FA35PbI3_MA50FA50PbI3\PL_PC_T_Cs30MA35FA35PbI3_MA50FA50PbI3_1sun_65C_40RH_air'


# Compute FFT stats or not
fft_stats = False


def _calc_binary_mask(primary_vid, mask):
    img = primary_vid[0, mask]
    classif = GaussianMixture(n_components=2)
    classif.fit(img[:, None])
    binary_img = classif.predict(img[:, None])
    bin_ = np.zeros_like(primary_vid[0])
    bin_[mask] = binary_img

    # Ensure than the ROI is True
    size_x = primary_vid.shape[1]
    window = size_x // 10
    mid_low, mid_high = size_x // 2 - window, size_x // 2 + window
    if bin_[mid_low:mid_high, mid_low:mid_high].mean() < 0.8:
        binary_img = ~binary_img+2

    return binary_img


# function for creating secondary videos
def _animate_vids(vids, cbar_label, vmin=None, vmax=None, text_list=None):
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    cv0 = vids[0]
    im = ax.imshow(cv0) # Here make an AxesImage rather than contour
    cb = fig.colorbar(im)
    cb.set_label(cbar_label)
    if text_list is not None:
        tx = ax.text(10, 30, text_list[0], Color=[1,1,1])

    def animate(fr):
        arr = vids[fr]

        if vmin is None:
            vmin_ = arr.min()
        else:
            vmin_ = vmin
        if vmax is None:
            vmax_ = arr.max()
        else:
            vmax_ = vmax
        im.set_data(arr)
        im.set_clim(vmin_, vmax_)
        #tx.set_text(text_list[fr])
        # In this version you don't have to do anything to the colorbar,
        # it updates itself when the mappable it watches (im) changes

    return animation.FuncAnimation(fig, animate, frames=len(vids))

# function for calculating temporal coefficient of variation of PLQY
def _plot_cv(plqy, t=None):
    def _get_slope_(log_x, log_y):
        A = np.vstack([log_x, np.ones_like(log_x)]).T
        mmm, ccc = np.linalg.lstsq(A, log_y, rcond=None)[0]
        return mmm, ccc

    is_masked = np.ma.is_masked(plqy)

    if is_masked:
        xx = plqy.mean(0).compressed()
    else:
        xx = plqy.mean(0).flatten()
    valid_xx = xx > 0
    
    #print(xx)
    log_x = np.log(xx[valid_xx])
    if is_masked:
        log_y = np.log(plqy.std(0).compressed()[valid_xx])
    else:
        log_y = np.log(plqy.std(0).flatten()[valid_xx])

    mmm, ccc = _get_slope_(log_x, log_y)
    
    plt.scatter(log_x, log_y, s=1)
    xxline = np.percentile(log_x, [.5, 98])
    plt.plot(xxline, mmm * xxline + ccc, c=sns.color_palette()[1], lw=3)


    if t is not None:
        title = 'CV plot at t = %i, slope = %.2f' % (t, mmm)
    else:
        title = 'CV plot, slope = %.2f' % mmm

    plt.title(title)
    plt.ylabel('$\\log(\\mathrm{std}(\\mathrm{PLQY})_{T_P})$')
    plt.xlabel('$\\log(\\langle\\mathrm{PLQY}\\rangle_{T_P})$')

    return mmm


def _inverse_transform(array, valid_indices, allnum):
    if array.ndim > 1:
        new_array = np.zeros((allnum, ) + array.shape[1:])
    else:
        new_array = np.zeros(allnum)
    new_array[valid_indices] = array
    return new_array

# analytical function describing solar cell I-V curves
def IV_Lambert(V,Rs,Rsh,n_id,temp,Isc,Voc):
    # Variables:
    # V - voltage

    # Unknown Parameters:
    # Rs - series resistance
    # Rsh - shunt resistance
    # n_id - diode ideality factor

    # Known Parameters:
    # temp - Kelvin temperature
    # Isc - short-circuit current
    # Voc - open-circuit voltage
    
    k_B = const.kb # Boltzmann's constant, J/K
    qe = const.q # elementary charge, C

    # condensed parameter groups to make things easier
    V_T = k_B*temp/qe # thermal voltage
    alpha = 1 - np.exp((Rs*Isc-Voc)/(n_id*V_T))
    beta = (Isc + (Rs*Isc-Voc)/Rsh)/alpha
    gamma = Rs*(beta + Voc/Rsh)

    # current through the solar cell
    I = V/Rs - (Rsh/(Rs*(Rs + Rsh)))*(gamma + V) + n_id*V_T/Rs *\
        sp.special.lambertw( Rs/(n_id*V_T) * (Isc - Voc/(Rs+Rsh)) * np.exp(-Voc/(n_id*V_T))/alpha *\
                             np.exp( Rsh/(Rs + Rsh) * (gamma + V)/(n_id*V_T) ),
                           )

    return np.real(I) # lambertw() spits out complex values by default

# function to convert Lumencor LED setting to total beam power based on a polynomial fit of arbitrary degree
def Lumencor_to_watts(setting,coeffs,responsivity=0.3586):
    # setting: Lumencor LED setting on 255 scale           
    # coeffs: list of coefficients of polynomial fit, in order of descending degree
    # responsivity: convert photodiode readings from A to W (default is 0.3586 A/W at ~550 nm)
    #               if coefficients are already expressed in terms of power, set responsivity = 1
    b_power = 0
    for i in range(len(coeffs)):
        b_power += coeffs[i]*setting**(len(coeffs) - 1 - i)
    b_power /= responsivity
    return b_power


def main():
    
    
    # Automatically push raw videos to Drive

    #for directory in directory_list:
    #    experiment_info = utils.load_sample_metadata(directory, os.path.join('Experiment Info','experiment_info.json'))
    #    dest_raw =  os.path.join(
    #        #'drive:Machine_Learning/Data/Timeseries/', experiment_info['ClassID'], OLD RYAN METHOD
    #        #'gdrive:Machine_Learning/Data/Timeseries/', experiment_info['ClassID'], #Effort_Perovskties_2; this filled up in mid October 2020
    #        'gdrive4:Effort_Perovskites_4/Machine_Learning/Data/Timeseries/', #Links to a folder in Tim Siegler's Drive. To reconfigure a new drive, open cmd and run "rclone config" and follow instructios
    #        experiment_info['ClassID'],
    #        experiment_info['ExperimentID'], 'primary_vids')
    #    source_raw = directory

    #    command = (['rclone', 'copy', source_raw, dest_raw])
    #    subprocess.Popen(command)

    
    # Main loop
    for d_idx, directory in enumerate(directory_list):
        # Define the list of files to be analyzed
        ReadDir = directory
        print("\n-------------------------------------")
        print(ReadDir)
        sample_info = utils.load_sample_metadata(directory, os.path.join('Sample Info','sample_info.json'))
        experiment_info = utils.load_sample_metadata(directory, os.path.join('Experiment Info','experiment_info.json'))
        
        # get Kelvin temperature
        T_K = experiment_info['Temperature (deg C)'] + 273
        
        # determine whether sample is a film or a device
        try:
            sample_type = experiment_info['Sample Type']
        except:
            sample_type = 'film'
        
        # get device active area if applicable
        if sample_type == 'device':
            try: 
                active_area = sample_info['active_area, cm2']
            except:
                active_area = 0.06
            # assign the region of interest if using a device
            dev_ROI_x = sample_info['device_ROI_x']
            dev_ROI_y = sample_info['device_ROI_y']
        # determine which microscope was used to collect data
        # default is microscope 1
        try: 
            microscope = experiment_info['Microscope']
        except: 
            microscope = 1
        
        # determine which objective was used to collect data
        # default is Mitutoyo_100X
        try: 
            objective = experiment_info['Objective']
        except:
            objective = 'Mitutoyo_100X'
            
        # get the illumination wavelength/photon energy based on which microscope was used
        if microscope == 1:
            wavelength = 560e-9
        elif microscope == 2:
            wavelength = 550e-9
        else:
            print('Invalid microscope label:',microscope)
        E_photon = const.h*const.c/wavelength
            
        # construct string specifying neutral density filter combination
        # default is ND06 in, others out
        try:
            ND06_status = experiment_info['ND06']
            ND25_status = experiment_info['ND25']
            ND50_status = experiment_info['ND50']
            filter_string = ND06_status + '-' + ND25_status + '-' + ND50_status
        except: 
            filter_string = 'in-out-out'
        
        # get sample band gap
        # default is 1.61 eV
        try: 
            bandgap = sample_info['Band Gap (eV)']
        except:
            bandgap = 1.61
        
        
        # use microscope optics configuration to load the right power settings
        # dictionary
        optics_config_dir = 'Beam Intensity Calibration/'
        config_name = 'MS' + str(microscope) + '_' + objective + '_' + str(bandgap) + '_eV_Lumen255_to_Nsuns.json'
        config_path = os.path.join(optics_config_dir,config_name)
        
        try:
            with open(config_path,'r') as json_file:
                optics_dict = json.load(json_file)
            
            # extract the parameters governing the lumencor setting - Nsuns conversion
            # for a given filter setting
            try:
                sun_params = optics_dict[filter_string]
            except KeyError:
                print('Invalid Filter Combination')
                break
            # load the PL calibration factor (default is characteristic of microscope 1,
            # Mitutoyo 100X objective)
            try:
                PL_cal_fac = optics_dict['PL_cal_factor']
            except:
                PL_cal_fac = 1.7832362718915656e14
        except:
            yes_or_no = input('Valid optics configuration file not found. Continue anyway? Input "y" to continue, "n" to break out of loop: ')
            if yes_or_no == 'n':
                break
        
            
        # determine whether dark field videos were taken
        # default is to assume they weren't
        try:
            dark_field = experiment_info['Dark_Field']
        except:
            dark_field = False

        # determine whether bright field videos were taken
        # default is to assume they weren't
        try:
            bright_field = experiment_info['Bright_Field']
        except:
            bright_field = False   
        
        
        try:
            n_grads = experiment_info['nGradients']
            n_g_locs = experiment_info['nGradPoints']
        except KeyError:
            n_grads = 1
            n_g_locs = 1

        def _get_sun_contact():
            NSuns = experiment_info['Excitation Intensity']
            Ld_contacts = experiment_info['Ld_contacts']
            using_XYstage = experiment_info['using_XYstage']
            return NSuns, Ld_contacts, using_XYstage

        NSuns, Ld_contacts, using_XYstage = _get_sun_contact()
         
        
        for grad in range(n_grads):
            for g_loc in range(n_g_locs):
                if n_g_locs < 2:
                    WriteDir = os.path.join('Output/',
                                        experiment_info['ClassID'], experiment_info['ExperimentID'])
                else:
                    WriteDir = os.path.join('Output/',
                                        experiment_info['ClassID'], experiment_info['ExperimentID'],
                                        'grad' + str(grad), 'loc' + str(g_loc))
                # WriteDir = os.path.expanduser('~/notebooks/research_notebook/PL_data_analysis/script_fix/temp_dir/')


                if not os.path.exists(WriteDir):
                    os.makedirs(WriteDir)

                
                # Conduct Ld and T calculations if measured
                
                if sample_type == 'film' and experiment_info['Ld_data']:
                    # return not only LD data but also the Lumencor power setting for BF/DF calculations
                    Ld_df, sun_setting = Ld_analysis_batch.electronic_data_calculations(directory,
                                                                           WriteDir,
                                                                           experiment_info['Temperature (deg C)'] + 273,
                                                                           sample_info['Film Thickness, nm']*1e-9,
                                                                           experiment_info['channel_length'],
                                                                           experiment_info['channel_width'],
                                                                           experiment_info['vidTimeIntervalMs']/60000,
                                                                           experiment_info['Transmissivity_data'],
                                                                           sun_params,
                                                                           grad, g_loc)
                
                # this statement loads the primary videos into a 4D array
                primary_videos, valid_indices_primary = load_primary_videos(ReadDir, grad=grad, loc=g_loc)
                #primary_videos, valid_indices_primary = load_primary_videos(ReadDir, tiff_name='scaled_MMStack_Pos0.ome.tif', grad=grad, loc=g_loc)
                
                
                if microscope == 2:
                    exp_times, valid_indices_exp = load_exposure(ReadDir, grad=grad, loc=g_loc, exposure_keyname='pco_camera-Exposure')
                else:
                    exp_times, valid_indices_exp = load_exposure(ReadDir, grad=grad, loc=g_loc)
                #exp_times=exp_times/16 #if we had the wrong binning in initial videos ONLY; VERY RARE TO HAVE UNCOMMENTED
                valid_indices = np.logical_and(valid_indices_primary, valid_indices_exp)

                #exp_times = exp_times[valid_indices_primary[valid_indices_exp]]
                primary_videos = primary_videos[valid_indices_exp[valid_indices_primary]]

                numfiles = valid_indices.shape[0]
                num_valid_files, name_len1 = primary_videos.shape[:2]
                numFrames = experiment_info['nrFrames']


                exposure_times = exp_times[valid_indices]
                pixel_per_um_each_pv = utils.load_pixel_per_um(ReadDir)
                pixel_per_um = pixel_per_um_each_pv.mean()
                
                
                if not np.allclose(pixel_per_um_each_pv, pixel_per_um):
                    raise ValueError('Pixel per um not the same for different primary videos')

                dark_vid_count = 0

                two_d_window = generate_2d_round_window(primary_videos.shape[-1], .4)

                # Pre-allocate vectors for statistics
                xy0t0 = np.zeros([numfiles-dark_vid_count,1]) #grand video average vs time
                xy0t0_cts = np.zeros([numfiles-dark_vid_count,1]) #average in raw counts
                xy1t0 = np.zeros([numfiles-dark_vid_count,1]) #spatial standard deviation of time average
                xy2t0 = np.zeros([numfiles-dark_vid_count,1]) #spatial skewness of time average
                xy3t0 = np.zeros([numfiles-dark_vid_count,1]) #spatial kurtosis of time average
                xy0t1 = np.zeros([numfiles-dark_vid_count,1]) #spatial average of the time STD image
                xy1t1 = np.zeros([numfiles-dark_vid_count,1]) #spatial STD of the time STD image
                xy2t1 = np.zeros([numfiles-dark_vid_count,1]) #spatial skewness of time STD
                xy3t1 = np.zeros([numfiles-dark_vid_count,1]) #spatial kurtosis of time STD
                xy0t1Norm = np.zeros([numfiles-dark_vid_count,1]) #Normalize each time STD pixel to its time average
                t0xy1 = np.zeros([numfiles-dark_vid_count,1]) #take the spatial average first, then STD of t values
                t1xy0 = np.zeros([numfiles-dark_vid_count,1]) #Take the spatial ave of each image and then take the std of those
                t0xy1 = np.zeros([numfiles-dark_vid_count,1]) #Take the spatial std of each image and then take the ave of those
                Bbig1 = np.zeros([numfiles-dark_vid_count,1]) #Standard deviation of the slope parameter
                PLQY_xy0t0 = np.zeros([numfiles-dark_vid_count,1]) #spatial average PLQY vs time
                QFLS_xy0t0 = np.zeros([numfiles-dark_vid_count,1]) #spatial average PLQY vs time
                exp_times = np.zeros([numfiles-dark_vid_count,1])
                uq_distance = np.zeros([numfiles-dark_vid_count,1])

                beta_mean_xy_vs_t = np.zeros([numfiles - dark_vid_count])
                beta_std_xy_vs_t = np.zeros([numfiles - dark_vid_count])
                frac_bright = np.zeros([numfiles - dark_vid_count])
                cv_slopes = np.zeros([numfiles - dark_vid_count])

                # Pre-allocate lists for export videos
                ims_std = []
                ims_mean = []
                ims_chi = []
                betas = []
                plqy_all = []

                # for scaling video:
                chi_max = 0
                chi_min = 1


                #calculation

                vid_idxs = valid_indices.nonzero()[0]
                NSuns_list = (np.ones(numfiles) * NSuns)[valid_indices]
                Incident_flux = const.calc_incident_flux(NSuns_list)


                est_rw_x, est_rw_y = utils.estimate_movement(primary_videos)

                t = np.array([experiment_info['vidTimeIntervalMs'] / 1000 * ii
                              for ii in range(numfiles)])
                binary_masks = []
                is_masked = False
                
                
                # preallocate lists for device data and determine its structure
                if sample_type == 'device':
                    # look at first cycle and find out which measurement modules exist
                    start_data = np.loadtxt(os.path.join(directory, 'electronic_measurement_data_cycle0.csv'),delimiter=',')
                    status_column = start_data[:,8]
                    
                    
                    # initialize data flags and lists
                    Voc_collected = False
                    Voc_data = []
                    MPPT_collected = False
                    MPPT_data = []
                    Isc_collected = False
                    Isc_data = []
                    Light_IV_collected = False
                    Light_IV_data = []
                    Dark_IV_collected = False
                    Dark_IV_data = []
                    
                    # determine which data collection regimes were active
                    if 1 in status_column:
                        Voc_collected = True    
                    if 2 in status_column:
                        MPPT_collected = True
                    if 3 in status_column:
                        Isc_collected = True
                    if 4 in status_column:
                        Light_IV_collected = True
                    if 5 in status_column:
                        Dark_IV_collected = True
                
                if sample_type == 'device':
                    jv_datafiles = natsorted([name for name in os.listdir(directory) if 'electronic_measurement_data_cycle' in name])
                    for ii,jv_file in enumerate(jv_datafiles):
                        # load data from the appropriate cycle
                        jv_data = np.loadtxt(os.path.join(directory,jv_file),delimiter=',')
        
                        # Break data down into separate collection regimes
                        if Voc_collected:
                            Voc_data.append(jv_data[np.where(status_column==1)[0],:8])                        
                        if MPPT_collected:
                            MPPT_data.append(jv_data[np.where(status_column==2)[0],:8])                        
                        if Isc_collected:
                            Isc_data.append(jv_data[np.where(status_column==3)[0],:8])                        
                        if Light_IV_collected:
                            Light_IV_data.append(jv_data[np.where(status_column==4)[0],:8])                    
                        if Dark_IV_collected:
                            Dark_IV_data.append(jv_data[np.where(status_column==5)[0],:8])

                
                
                # Main loop, loop over number of short videos in video series
                for ii, (primary_vid, exposure_time, NSun, vid_idx, inc_flux, mx, my) in tqdm(enumerate(zip(
                    primary_videos, exposure_times, NSuns_list, vid_idxs, Incident_flux, est_rw_x, est_rw_y)),
                    total=num_valid_files, desc='Main loop'):
    
    
                    # analyze device data, if applicable

                    # analyze PL videos
                    primary_vid_ = utils.cancel_movement_per_primary_vid(
                        primary_vid, mx, my)
                    mask_move = primary_vid_[0] != 0.0
                    if np.all(mask_move) and not Ld_contacts:
                        binary_mask = None
                        binary_masks.append(np.ones_like(mask_move, dtype=bool))
                    else:
                        is_masked = True
                        binary_mask = np.zeros_like(primary_vid_[0], dtype=bool)
                        if Ld_contacts: 
                            binary_mask[mask_move] = _calc_binary_mask(primary_vid_, mask_move)
                        else:
                            binary_mask[mask_move] = True
                        binary_masks.append(binary_mask)

                    Array = calibrate_videos(primary_vid_, exposure_time, binary_mask, cal_fac = PL_cal_fac)
                    if ii == 0:
                        x_dim = Array.shape[1]
                        y_dim = Array.shape[2]
                        t_dim = Array.shape[0]
                        
                    #TODO: enfore device region of interest
                    if sample_type == 'device':
                        Array = Array[:,dev_ROI_x[0]:dev_ROI_x[1],dev_ROI_y[0]:dev_ROI_y[1]]

                    plqy_im, mean_im = calc_plqy_qfls(Array, NSun)
                    plqy_vids = Array / inc_flux

                    stdev_im = np.std(Array, axis=0) / np.mean(Array) # divide by "global mean"
                    chi_im = mean_im / SQ_calcs.VocMax(const.sample_bandgap, NSuns)
                    try:
                        chi_max = np.max([np.max(chi_im), chi_max])
                        chi_min = np.min([np.min(chi_im), chi_min])
                    except ValueError:
                        print('Warning ValueError')
                    beta = utils.compute_photobrightening(Array[None, ...], axis=1)[0].squeeze()

                    # images in videos
                    ims_std.append(stdev_im)
                    ims_mean.append(mean_im)
                    ims_chi.append(chi_im)
                    betas.append(beta)
                    plqy_all.append(plqy_im)


                    # Calculate features

                    try:
                        xy0t0[vid_idx] = Array.mean()
                        xy0t0_cts[vid_idx] = xy0t0[vid_idx]*exposure_time
                        xy1t0[vid_idx] = Array.mean(axis=0).std()
                        array_mean0 = Array.mean(axis=0)
                        #array_mean0 = array_mean0.compressed() if is_masked else array_mean0.flatten()
                        array_mean0 = array_mean0.flatten()
                        xy2t0[vid_idx] = skew(array_mean0)
                        xy3t0[vid_idx] = kurtosis(array_mean0)

                        xy0t1[vid_idx] = Array.std(axis=0).mean()
                        xy1t1[vid_idx] = Array.std(axis=0).std()

                        array_std0 = Array.std(axis=0)
                        #array_std0 = array_std0.compressed() if is_masked else array_std0.flatten()
                        array_std0 = array_std0.flatten()
                        xy2t1[vid_idx] = skew(array_std0)
                        xy3t1[vid_idx] = kurtosis(array_std0)

                        xy0t1Norm[vid_idx] = (Array.std(axis=0)/Array.mean(axis=0)).mean()
                        t0xy1[vid_idx] = Array.std(axis=1).mean()
                        t1xy0[vid_idx] = Array.mean(axis=1).std()
                        PLQY_xy0t0[vid_idx] = xy0t0[vid_idx]/inc_flux
                        QFLS_xy0t0[vid_idx] = SQ_calcs.VocMax(const.sample_bandgap, NSun) + \
                            const.keV*const.T*np.log(PLQY_xy0t0[vid_idx])

                        # beta_ = beta.flatten()
                        beta_mean_xy_vs_t[vid_idx] = beta.mean()
                        beta_std_xy_vs_t[vid_idx] = beta.std()
                        #beta_flat = beta.compressed() if is_masked else beta.flatten()
                        beta_flat = beta.flatten()
                        frac_bright[vid_idx] = np.mean(beta_flat > 0)
                    except ValueError:
                        print('Warning ValueError')


                    plt.figure()
                    try:
                        cv_slopes[vid_idx] = _plot_cv(Array / inc_flux, t=t[vid_idx])
                    except:
                        cv_slopes[vid_idx] = np.nan
                    cv_dir = os.path.join(WriteDir, 'cv_plots')
                    if not os.path.exists(cv_dir):
                        os.makedirs(cv_dir)
                    plt.savefig(os.path.join(cv_dir, '%03i.png' % vid_idx), dpi=300,
                                bbox_inches='tight')
                    plt.close()

                    if fft_stats:
                        # TODO: how to deal with invalid points?
                        fft_mean_vid_windowed = np.fft.fftshift(np.fft.fft2(
                            plqy_vids[::5] * two_d_window[None, ...]), axes=(1, 2))

                        radi_angle_array, size_array, radi_range, angle_range = \
                            generate_radii_angle_array(fft_mean_vid_windowed)

                        mean_fft_spectrum = np.ma.mean(np.ma.log10(radi_angle_array), axis=2).data
                        _until = math.ceil(mean_fft_spectrum.shape[1] / np.sqrt(2))
                        coord_x_max = _until / (pixel_per_um * primary_videos.shape[-1])
                        xxx_radi = np.linspace(0, coord_x_max, _until)
                        plt.figure()
                        for i, ix in enumerate(range(0, numFrames, 5)):
                            plt.plot(xxx_radi, mean_fft_spectrum[i, :_until].T,
                                     label=('$T_P = %02i$' % (ix+1)).replace('0', ' '))
                        plt.xlabel('reciprocal space $s$ [$\\mu m^{-1}$]')
                        plt.ylabel('$\\log(F)$')
                        plt.legend(ncol=2, framealpha=0.0)
                        fft_dir = os.path.join(WriteDir, 'fft_plots')
                        if not os.path.exists(fft_dir):
                            os.makedirs(fft_dir)

                        plt.savefig(os.path.join(fft_dir, '%03i.png' % vid_idx), dpi=300, bbox_inches='tight')
                        plt.close()

                # if specified, extract dark field features:
                if dark_field:
                    # first identify the image files
                    if sample_type == 'film':
                        DF_names = natsorted([name for name in os.listdir(directory) 
                                              if 'grad' + str(grad) in name 
                                              and 'loc' + str(g_loc) in name
                                              and 'DF_Exp' in name
                                              ])
                    if sample_type == 'device':
                        DF_names = natsorted([name for name in os.listdir(directory) 
                                              if 'grad' + str(grad) in name 
                                              and 'loc' + str(g_loc) in name
                                              and ('DF_exp' in name or 'DF_Exp' in name)
                                              ])
                        
                            
                    # specify DF region of interest (for example, if concerned about vignetting)
                    pix_size = 512 # wi\dth of image, in pixels
                    DF_ROI_x = [int(pix_size/4),int(pix_size*3/4)]
                    DF_ROI_y = [int(pix_size/4),int(pix_size*3/4)]
                    # if sample is a device, use the assigned ROI
                    if sample_type == 'device':
                        DF_ROI_x = dev_ROI_x
                        DF_ROI_y = dev_ROI_y
                    
                    # Define DF features
                    DF_means = np.zeros([len(DF_names),1]) # mean
                    DF_medians = np.zeros([len(DF_names),1]) # medians
                    DF_stds = np.zeros([len(DF_names),1]) # std deviation
                    DF_skews = np.zeros([len(DF_names),1]) # skewness
                    DF_kurts = np.zeros([len(DF_names),1]) # kurtosis
                    
                    DF_ims = []
                    
                    # use the dark field images to calculate diffuse reflectance, if possible     
                    DF_config_dir = '../wf_pl/Beam Intensity Calibration/Bright Field'
                    DF_config_name = 'MS' + str(microscope) + '_' + objective + '_DF_config.json'
                    DF_config_path = os.path.join(DF_config_dir,DF_config_name)
                    DF_VF_name = 'MS' + str(microscope) + '_' + objective + '_DF_VarFcn.csv'
                    DF_VF_path = os.path.join(DF_config_dir,DF_VF_name)
                    try:
                        # first load the config file
                        with open(DF_config_path,'r') as json_file:
                            DF_dict = json.load(json_file)
                        # now load the variation function
                        DF_VarFcn = np.loadtxt(DF_VF_path,delimiter=' ')
                        # extract the beam power parameters
                        power_params = DF_dict[filter_string]
                        # calculate the beam power - set responsivity = 1 because already specified as beam power
                        DF_power = Lumencor_to_watts(sun_setting,power_params,responsivity=1)
                        # extract the field of view area
                        A_FOV = DF_dict['FOV Area']
                        A_px = A_FOV/pix_size**2
                        # extract the BF scaling factor
                        k_DF = DF_dict['Scaling Factor']
                        # note successful extraction of calibration parameters                        
                        DF_cal_exists = True 
                        # initialize specular reflectance stats arrays
                        Rdiff_means = np.zeros([len(DF_names),1]) # mean
                        Rdiff_medians = np.zeros([len(DF_names),1]) # medians
                        Rdiff_stds = np.zeros([len(DF_names),1]) # std deviation
                        Rdiff_skews = np.zeros([len(DF_names),1]) # skewness
                        Rdiff_kurts = np.zeros([len(DF_names),1]) # kurtosis
                        Rdiff_ims = []
                    except:
                        DF_cal_exists = False                    
                    
                    DF_bkg = 1600 # assume background counts = 1600 for 4x4 binning
                    for jj, DF_img in enumerate(DF_names):
                        DF_Array = plt.imread(directory + '/' + DF_img) # load DF image
                        if sample_type == 'film':
                            DF_exp = float(DF_img.split('Exp')[1].split('ms')[0])/1000 # get exposure time in sec
                        if sample_type == 'device':
                            try:
                                DF_exp = float(DF_img.split('exp')[1].split('.tif')[0])/1000
                            except:
                                DF_exp = float(DF_img.split('Exp')[1].split('ms')[0])/1000
                        DF_cps = DF_Array/DF_exp # convert image pixel intensity to counts per second
                        
                        # calculate DF statistics features
                        DF_means[jj] = np.mean(DF_cps[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                        DF_medians[jj] = np.median(DF_cps[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                        DF_stds[jj] = np.std(DF_cps[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                        DF_skews[jj] = skew(DF_cps[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]],axis=None)
                        DF_kurts[jj] = kurtosis(DF_cps[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]],axis=None)
                        
                        # append DF image for video production
                        DF_ims.append(DF_cps)
                        
                        # if valid config file exists, calculate specular reflectance too
                        if DF_cal_exists:
                            Rdiff_Array = k_DF*(DF_Array-DF_bkg)*E_photon/(DF_power*DF_exp*A_px*DF_VarFcn)
                            Rdiff_ims.append(Rdiff_Array)
                            Rdiff_means[jj] = np.mean(Rdiff_Array[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                            Rdiff_medians[jj] = np.median(Rdiff_Array[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                            Rdiff_stds[jj] = np.std(Rdiff_Array[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]])
                            Rdiff_skews[jj] = skew(Rdiff_Array[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]],axis=None)
                            Rdiff_kurts[jj] = kurtosis(Rdiff_Array[DF_ROI_x[0]:DF_ROI_x[1],DF_ROI_y[0]:DF_ROI_y[1]],axis=None)
                        
                # if specified, extract bright field features:                        
                if bright_field:
                    # first identify the image files
                    BF_names = natsorted([name for name in os.listdir(directory) 
                                          if 'grad' + str(grad) in name 
                                          and 'loc' + str(g_loc) in name
                                          and 'BF_Exp' in name
                                          ])
                    # specify BF region of interest (for example, if concerned about vignetting)
                    pix_size = 512
                    BF_ROI_x = [int(pix_size/4),int(pix_size*3/4)]
                    BF_ROI_y = [int(pix_size/4),int(pix_size*3/4)]
                    # if sample is a device, use the assigned ROI
                    if sample_type == 'device':
                        BF_ROI_x = dev_ROI_x
                        BF_ROI_y = dev_ROI_y
                        
                    # Define BF features
                    BF_means = np.zeros([len(DF_names),1]) # mean
                    BF_medians = np.zeros([len(DF_names),1]) # medians
                    BF_stds = np.zeros([len(DF_names),1]) # std deviation
                    BF_skews = np.zeros([len(DF_names),1]) # skewness
                    BF_kurts = np.zeros([len(DF_names),1]) # kurtosis
                    
                    BF_ims = []
                    
                    # use the bright field images to calculate specular reflectance, if possible     
                    BF_config_dir = '../wf_pl/Beam Intensity Calibration/Bright Field'
                    BF_config_name = 'MS' + str(microscope) + '_' + objective + '_BF_config.json'
                    BF_config_path = os.path.join(BF_config_dir,BF_config_name)
                    BF_VF_name = 'MS' + str(microscope) + '_' + objective + '_BF_VarFcn.csv'
                    BF_VF_path = os.path.join(BF_config_dir,BF_VF_name)
                    try:
                        # first load the config file
                        with open(BF_config_path,'r') as json_file:
                            BF_dict = json.load(json_file)
                        # now load the variation function
                        BF_VarFcn = np.loadtxt(BF_VF_path,delimiter=' ')
                        
                        # extract the beam power parameters
                        power_params = BF_dict[filter_string]
                        # calculate the beam power - set responsivity = 1 because already specified as beam power
                        BF_power = Lumencor_to_watts(sun_setting,power_params,responsivity=1)
                        # extract the field of view area
                        A_FOV = BF_dict['FOV Area']
                        A_px = A_FOV/pix_size**2
                        # extract the fraction of total beam power that is in the FOV
                        align_factor = BF_dict['FOV Beam Power Fraction']
                        # calculate the average irradiance in the FOV (as photon flux)
                        irradiance_ph = BF_power*align_factor/A_FOV / E_photon
                        # extract the BF scaling factor
                        k_BF = BF_dict['Scaling Factor']
                        # note successful extraction of calibration parameters                        
                        BF_cal_exists = True 
                        # initialize specular reflectance stats arrays
                        Rspec_means = np.zeros([len(BF_names),1]) # mean
                        Rspec_medians = np.zeros([len(BF_names),1]) # medians
                        Rspec_stds = np.zeros([len(BF_names),1]) # std deviation
                        Rspec_skews = np.zeros([len(BF_names),1]) # skewness
                        Rspec_kurts = np.zeros([len(BF_names),1]) # kurtosis
                        Rspec_ims = []
                    except:
                        BF_cal_exists = False
                    
                    BF_bkg = 1600 # assume background is ~ 1600 counts in 4x4 binning
                    for jj, BF_img in enumerate(BF_names):
                        
                        BF_Array = plt.imread(directory + '/' + BF_img) # load DF image
                        BF_exp = float(BF_img.split('Exp')[1].split('ms')[0])/1000 # get exposure time in sec
                        BF_cps = (BF_Array-BF_bkg)/BF_exp # convert image pixel intensity to counts per second
                        
                        # calculate BF statistics features
                        BF_means[jj] = np.mean(BF_cps[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                        BF_medians[jj] = np.median(BF_cps[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                        BF_stds[jj] = np.std(BF_cps[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                        BF_skews[jj] = skew(BF_cps[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]],axis=None)
                        BF_kurts[jj] = kurtosis(BF_cps[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]],axis=None)  
                        
                        # append BF image for video production
                        BF_ims.append(BF_cps)
                        
                        # if valid config file exists, calculate specular reflectance too
                        if BF_cal_exists:
                            Rspec_Array = k_BF*(BF_Array-BF_bkg)/(irradiance_ph*BF_exp*A_px*BF_VarFcn)
                            Rspec_ims.append(Rspec_Array)
                            Rspec_means[jj] = np.mean(Rspec_Array[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                            Rspec_medians[jj] = np.median(Rspec_Array[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                            Rspec_stds[jj] = np.std(Rspec_Array[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]])
                            Rspec_skews[jj] = skew(Rspec_Array[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]],axis=None)
                            Rspec_kurts[jj] = kurtosis(Rspec_Array[BF_ROI_x[0]:BF_ROI_x[1],BF_ROI_y[0]:BF_ROI_y[1]],axis=None)
                
                # analyze device data if applicable
                if sample_type == 'device':
                
                    # analyze quasi-stable Voc                    
                    if Voc_collected:
                        Voc_timeseries = np.zeros(len(Voc_data))
                        
                        # for each cycle, calculate Voc by averaging the last 10 collected data points
                        for ii in range(len(Voc_data)):
                            Voc_timeseries[ii] = np.mean(Voc_data[ii][-10:,6])
                    
                    # analyze maximum power point data
                    if MPPT_collected:
                        Vmpp_timeseries = np.zeros(len(MPPT_data))
                        Impp_timeseries = np.zeros(len(MPPT_data))
                        
                        # get voltage and current at MPP
                        for ii in range(len(MPPT_data)):
                            Vmpp_timeseries[ii] = np.mean(MPPT_data[ii][-10:,6])
                            Impp_timeseries[ii] = np.mean(MPPT_data[ii][-10:,4])
                    
                    # analyze quasi-stable Isc
                    if Isc_collected:
                        Isc_timeseries = np.zeros(len(Isc_data))
                        
                        # for each cycle, calculate Isc by averaging the last 10 collected data points
                        for ii in range(len(Isc_data)):
                            Isc_timeseries[ii] = np.mean(Isc_data[ii][-10:,6])
                            
                    # analyze light JV curves
                    if Light_IV_collected:
                        Isc_fwd_timeseries  = np.zeros(len(Light_IV_data))
                        Voc_fwd_timeseries  = np.zeros(len(Light_IV_data))
                        FF_fwd_timeseries   = np.zeros(len(Light_IV_data))
                        Rs_fwd_timeseries   = np.zeros(len(Light_IV_data))
                        Rsh_fwd_timeseries  = np.zeros(len(Light_IV_data))
                        n_id_fwd_timeseries = np.zeros(len(Light_IV_data))
                        I0_fwd_timeseries   = np.zeros(len(Light_IV_data))
                        IL_fwd_timeseries   = np.zeros(len(Light_IV_data))
                        R2_fwd_timeseries   = np.zeros(len(Light_IV_data))
                        Rs_fwd_timeseries_slope   = np.zeros(len(Light_IV_data))
                        Rsh_fwd_timeseries_slope  = np.zeros(len(Light_IV_data))
                        
                        Isc_rev_timeseries  = np.zeros(len(Light_IV_data))
                        Voc_rev_timeseries  = np.zeros(len(Light_IV_data))
                        FF_rev_timeseries   = np.zeros(len(Light_IV_data))
                        Rs_rev_timeseries   = np.zeros(len(Light_IV_data))
                        Rsh_rev_timeseries  = np.zeros(len(Light_IV_data))
                        n_id_rev_timeseries = np.zeros(len(Light_IV_data))
                        I0_rev_timeseries   = np.zeros(len(Light_IV_data))
                        IL_rev_timeseries   = np.zeros(len(Light_IV_data))
                        R2_rev_timeseries   = np.zeros(len(Light_IV_data))
                        Rs_rev_timeseries_slope   = np.zeros(len(Light_IV_data))
                        Rsh_rev_timeseries_slope  = np.zeros(len(Light_IV_data))
                        
                        for ii in range(len(Light_IV_data)):
                            Vsweep = Light_IV_data[ii][:,6]
                            Isweep = Light_IV_data[ii][:,4]
                            # break the data into forward and reverse sweeps
                            for jj in range(len(Vsweep)):
                                if (Vsweep[jj] == Vsweep[jj-1]) and (Vsweep[jj+1] > Vsweep[jj]):
                                    turning_point = jj
                            V_fwd = Vsweep[turning_point:]
                            V_rev = Vsweep[:turning_point]
                            I_fwd = Isweep[turning_point:]
                            I_rev = Isweep[:turning_point]
                            
                            # calculate short circuit current
                            Isc_fwd_timeseries[ii] = I_fwd[0]
                            Isc_rev_timeseries[ii] = I_rev[-1]
                            Isc_fwd = I_fwd[0]
                            Isc_rev = I_rev[-1]
                            
                            # calculate Voc
                            for jj in range(len(I_fwd)-1):
                                if I_fwd[jj] < 0 and I_fwd[jj+1] > 0:
                                    Voc_fwd_timeseries[ii] = V_fwd[jj] + (V_fwd[jj+1] - V_fwd[jj])/(I_fwd[jj+1] - I_fwd[jj])*(0 - I_fwd[jj])
                                    Voc_fwd = Voc_fwd_timeseries[ii]
                                    break
                            #print('Debug: In Cycle #',ii)

                                    
                            # calculate fill factor
                            power_fwd = I_fwd*V_fwd
                            fwd_MPP_idx = np.argmin(power_fwd)
                            FF_fwd_timeseries[ii] = I_fwd[fwd_MPP_idx]*V_fwd[fwd_MPP_idx]/(Voc_fwd_timeseries[ii]*Isc_fwd_timeseries[ii])
                            
                                                        
                            # now fit curves using the Lambert W-function based solution:
                                
                            # forward first:
                            # estimate the series and shunt resistances from curve slopes
                            # (abs. value because sometimes the slopes are negative)
                            Rsh_guess = (V_fwd[1] - V_fwd[0])/((I_fwd[1] - I_fwd[0]))
                            Rs_guess = (V_fwd[jj+1] - V_fwd[jj])/((I_fwd[jj+1] - I_fwd[jj]))
                            Rs_fwd_timeseries_slope[ii]   = Rs_guess
                            Rsh_fwd_timeseries_slope[ii]  = Rsh_guess
        
                            # define a helper function for fitting
                            def helper(V,Rs,Rsh,n_id):
                                return IV_Lambert(V, Rs, Rsh, n_id, T_K, -Isc_fwd, Voc_fwd)
                            # for a rare but troublesome minority of points, the fit fails - in these cases, 
                            # just assign NaNs to the fit parameters
                            try:
                                # fit the curve using known temperature, Isc, and Voc 
                                popt_fwd,pcov_fwd = sp.optimize.curve_fit(helper, V_fwd, I_fwd, p0 = [Rs_guess,Rsh_guess,2], maxfev=100000, bounds=([0,0,1],[np.inf,np.inf,10]))   
                                # report the parameters
                                Rs_fwd, Rsh_fwd, n_id_fwd = popt_fwd
                                # calculate other important parameters
                                V_T = kB*T_K/q # thermal voltage
                                alpha_fwd = 1 - np.exp((Rs_fwd*Isc_fwd-Voc_fwd)/(n_id_fwd*V_T))
                                beta_fwd = (Isc_fwd + (Rs_fwd*Isc_fwd-Voc_fwd)/Rsh_fwd)/alpha_fwd
                                I0_fwd = beta_fwd*np.exp(-Voc_fwd/(n_id_fwd*V_T)) # reverse saturation current
                                I_L_fwd = beta_fwd + Voc_fwd/Rsh_fwd # photocurrent
                                
                                # calculate fit R2
                                r2_fwd = r2_score(I_fwd,helper(V_fwd,Rs_fwd,Rsh_fwd,n_id_fwd))
                            except:
                                Rs_fwd = np.nan
                                Rsh_fwd = np.nan
                                n_id_fwd = np.nan
                                I0_fwd = np.nan
                                I_L_fwd = np.nan
                                r2_fwd = np.nan
                                
                            # now get reverse scan device parameters
                            
                            # Voc
                            for jj in range(len(I_rev)-1):
                                if I_rev[jj] > 0 and I_rev[jj+1] < 0:
                                    Voc_rev_timeseries[ii] = V_rev[jj] + (V_rev[jj+1] - V_rev[jj])/(I_rev[jj+1] - I_rev[jj])*(0 - I_rev[jj])
                                    Voc_rev = Voc_rev_timeseries[ii]
                                    break
                            # fill factor
                            power_rev = I_rev*V_rev
                            rev_MPP_idx = np.argmin(power_rev)
                            FF_rev_timeseries[ii] = I_rev[rev_MPP_idx]*V_rev[rev_MPP_idx]/(Voc_rev_timeseries[ii]*Isc_rev_timeseries[ii])

                            # estimate the series and shunt resistances from curve slopes
                            # (abs. value because sometimes the slopes are negative)
                            Rsh_guess = (V_rev[-2] - V_rev[-1])/(I_rev[-2] - I_rev[-1])
                            Rs_guess = (V_rev[jj] - V_rev[jj+1])/(I_rev[jj] - I_rev[jj+1])
                            Rs_rev_timeseries_slope[ii]   = Rs_guess
                            Rsh_rev_timeseries_slope[ii]  = Rsh_guess
        
                            # define a helper function for fitting
                            def helper(V,Rs,Rsh,n_id):
                                return IV_Lambert(V, Rs, Rsh, n_id, T_K, -Isc_rev, Voc_rev)
                            # for a rare but troublesome minority of points, the fit fails - in these cases, 
                            # just assign NaNs to the fit parameters
                            try:
                                # fit the curve using known temperature, Isc, and Voc 
                                popt_rev,pcov_rev = sp.optimize.curve_fit(helper, V_rev, I_rev, p0 = [Rs_guess,Rsh_guess,2], maxfev=100000, bounds=([0,0,1],[np.inf,np.inf,10]))   
                                # report the parameters
                                Rs_rev, Rsh_rev, n_id_rev = popt_rev
                                # calculate other important parameters
                                V_T = kB*T_K/q # thermal voltage
                                alpha_rev = 1 - np.exp((Rs_rev*Isc_rev-Voc_rev)/(n_id_rev*V_T))
                                beta_rev = (Isc_rev + (Rs_rev*Isc_rev-Voc_rev)/Rsh_rev)/alpha_rev
                                I0_rev = beta_rev*np.exp(-Voc_rev/(n_id_rev*V_T)) # reverse saturation current
                                I_L_rev = beta_rev + Voc_rev/Rsh_rev # photocurrent
                                
                                # calculate fit R2
                                r2_rev = r2_score(I_rev,helper(V_rev,Rs_rev,Rsh_rev,n_id_rev))
                            except:
                                Rs_fwd = np.nan
                                Rsh_fwd = np.nan
                                n_id_fwd = np.nan
                                I0_fwd = np.nan
                                I_L_fwd = np.nan
                                r2_fwd = np.nan
                            
                            Rs_fwd_timeseries[ii]   = Rs_fwd
                            Rsh_fwd_timeseries[ii]  = Rsh_fwd
                            n_id_fwd_timeseries[ii] = n_id_fwd
                            I0_fwd_timeseries[ii]   = I0_fwd
                            IL_fwd_timeseries[ii]   = I_L_fwd
                            R2_fwd_timeseries[ii]   = r2_fwd
                            
                            Rs_rev_timeseries[ii]   = Rs_rev
                            Rsh_rev_timeseries[ii]  = Rsh_rev
                            n_id_rev_timeseries[ii] = n_id_rev
                            I0_rev_timeseries[ii]   = I0_rev
                            IL_rev_timeseries[ii]   = I_L_rev
                            R2_rev_timeseries[ii]   = r2_rev
                        
                # sync the masks to have the same shape as the device ROI        
                if sample_type == 'device':
                    for ii in range(len(binary_masks)):
                        binary_masks[ii] = binary_masks[ii][dev_ROI_x[0]:dev_ROI_x[1],dev_ROI_y[0]:dev_ROI_y[1]]
                # Construct and save videos

                # additional features
                convert_arr_func = np.ma.array if is_masked else np.array
                ims_mean, ims_std, betas, ims_chi, plqy_all, binary_masks = map(
                    convert_arr_func, [ims_mean, ims_std, betas, ims_chi, plqy_all, binary_masks])

                # Save animations
                text_list = ['%d Sun' % nsun for nsun in NSuns_list]
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ani = _animate_vids(ims_std, '$t-std_{norm}$', .01, .1, text_list=text_list)

                writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                #ani.save(os.path.join(WriteDir, 't1_movie.mp4'), writer=writer)
                plt.close()

                fig2 = plt.figure()
                ax2 = fig2.add_subplot(111)
                ani2 = _animate_vids(ims_mean, '$\Delta E_F\ [eV]$', text_list=text_list)
                #ani2.save(os.path.join(WriteDir, 't0_qfls_movie.mp4'), writer=writer)
                plt.close()

                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111)
                ani3 = _animate_vids(ims_chi, '$\chi$', chi_min, chi_max, text_list=text_list)
                #ani3.save(os.path.join(WriteDir, 't0_chi_movie.mp4'), writer=writer)
                plt.close()

                fig4 = plt.figure()
                ax4 = fig4.add_subplot(111)
                ani4 = _animate_vids(betas, '$\\beta$', text_list=text_list)

                #ani4.save(os.path.join(WriteDir, 'beta_movie.mp4'), writer=writer)
                plt.close()


                fig5 = plt.figure()
                ax5 = fig5.add_subplot(111)
                ani5 = _animate_vids(
                    np.ma.divide(ims_mean, ims_std).filled(0),
                    '$\\langle \\mathrm{QLFS}\\rangle_{T_S} / \\mathrm{std}(\\mathrm{QLFS})_{T_S}$',
                    text_list=text_list)

                #ani5.save(os.path.join(WriteDir, 't0_qlfs_div_std.mp4'), writer=writer)
                plt.close()
                
                # make dark field video (with ROI superimposed) 
                if dark_field:
                    fig6 = plt.figure()
                    ax6 = fig6.add_subplot(111)
                    DF_x = [DF_ROI_x[0],DF_ROI_x[1],DF_ROI_x[1],DF_ROI_x[0],DF_ROI_x[0]]
                    DF_y = [DF_ROI_y[0],DF_ROI_y[0],DF_ROI_y[1],DF_ROI_y[1],DF_ROI_y[0]]
                    ax6.plot(DF_x,DF_y,'r--',linewidth=2)
                    ani6 = _animate_vids(DF_ims,
                        'Dark Field Intensity')
                    #ani6.save(os.path.join(WriteDir, 'DarkField.mp4'), writer=writer)
                    plt.close()
                    if DF_cal_exists:
                        fig6a = plt.figure()
                        ax6a = fig6a.add_subplot(111)
                        DF_x = [DF_ROI_x[0],DF_ROI_x[1],DF_ROI_x[1],DF_ROI_x[0],DF_ROI_x[0]]
                        DF_y = [DF_ROI_y[0],DF_ROI_y[0],DF_ROI_y[1],DF_ROI_y[1],DF_ROI_y[0]]
                        ax6a.plot(DF_x,DF_y,'r--',linewidth=2)
                        ani6a = _animate_vids(Rdiff_ims,
                            'Diffuse Reflectance',
                            text_list=text_list)
                        ani6a.save(os.path.join(WriteDir, 'R_Diffuse.mp4'), writer=writer)
                        plt.close()                       
                

                # make bright field video (with ROI superimposed)
                if bright_field:
                    fig7 = plt.figure()
                    ax7 = fig7.add_subplot(111)
                    BF_x = [BF_ROI_x[0],BF_ROI_x[1],BF_ROI_x[1],BF_ROI_x[0],BF_ROI_x[0]]
                    BF_y = [BF_ROI_y[0],BF_ROI_y[0],BF_ROI_y[1],BF_ROI_y[1],BF_ROI_y[0]]
                    ax7.plot(BF_x,BF_y,'r--',linewidth=2)
                    ani7 = _animate_vids(BF_ims,
                        'Bright Field Intensity',
                        text_list=text_list)
                    #ani7.save(os.path.join(WriteDir, 'BrightField.mp4'), writer=writer)
                    plt.close()
                    if BF_cal_exists:
                        fig7a = plt.figure()
                        ax7a = fig7a.add_subplot(111)
                        BF_x = [BF_ROI_x[0],BF_ROI_x[1],BF_ROI_x[1],BF_ROI_x[0],BF_ROI_x[0]]
                        BF_y = [BF_ROI_y[0],BF_ROI_y[0],BF_ROI_y[1],BF_ROI_y[1],BF_ROI_y[0]]
                        ax7a.plot(BF_x,BF_y,'r--',linewidth=2)
                        ani7a = _animate_vids(Rspec_ims,
                            'Specular Reflectance',
                            text_list=text_list)
                        ani7a.save(os.path.join(WriteDir, 'R_Specular.mp4'), writer=writer)
                        plt.close()
                        

                # make device videos
                if sample_type == 'device':
                    if Light_IV_collected:
                        fig8, ax8 = plt.subplots(1, 2, figsize=(20,10), squeeze=True)
    
                        vmin = np.min(ims_mean) # min QFLS to plot
                        vmax = np.max(ims_mean) # max QFLS to plot
                        QFLS_im = ims_mean[0]
                        im = ax8[1].imshow(QFLS_im, vmin=vmin, vmax=vmax)
                        ax8[1].set_yticklabels([])
                        ax8[1].set_xticklabels([])
                        im.set_clim(vmin, vmax)
                        cb = fig8.colorbar(im)
                        cb.set_label('$\Delta E_F\ [eV]$')
                        
                        def animate_JV_PL(fr):
                            QFLS_im = ims_mean[fr]
                            im.set_data(QFLS_im)
                            
                            # keep track of bad PL videos
                            bad_count = 0
                            for ii in range(fr):
                                if valid_indices[ii] == False:
                                    bad_count+=1
                            # increment to the index keeping track of JV data
                            fr = fr + bad_count
                            
                            # set up handles for data and fit
                            h1, = ax8[0].plot([], [], marker='o',markersize=12,linestyle=' ') # data
                            h2, = ax8[0].plot([], [], marker=' ',linewidth=2,linestyle='-') # fit
                            jvdata = Light_IV_data[fr]
                            
                            Vsweep = jvdata[:,6]
                            Isweep = jvdata[:,4]
                            
                            # break the data into forward and reverse sweeps
                            for jj in range(len(Vsweep)):
                                if (Vsweep[jj] == Vsweep[jj-1]) and (Vsweep[jj+1] > Vsweep[jj]):
                                    turning_point = jj
                                    break
                            V_rev = Vsweep[:turning_point]
                            I_rev = Isweep[:turning_point]
                            
                            # Recalculate reverse scan parameters from Lambert W-function fit
                            # Isc
                            Isc_rev = I_rev[-1]
                            # Voc
                            for jj in range(len(I_rev)-1):
                                if I_rev[jj] > 0 and I_rev[jj+1] < 0:
                                    Voc_rev = V_rev[jj] + (V_rev[jj+1] - V_rev[jj])/(I_rev[jj+1] - I_rev[jj])*(0 - I_rev[jj])
                                    break

                            # estimate the series and shunt resistances from curve slopes
                            # (abs. value because sometimes the slopes are negative)
                            Rsh_guess = np.abs((V_rev[-1] - V_rev[-2])/((I_rev[-1] - I_rev[-2])))
                            Rs_guess = np.abs((V_rev[jj+1] - V_rev[jj])/((I_rev[jj+1] - I_rev[jj])))
        
                            # define a helper function for fitting
                            def helper(V,Rs,Rsh,n_id):
                                return IV_Lambert(V, Rs, Rsh, n_id, T_K, -Isc_rev, Voc_rev)
                            
                            try:
                                # fit the curve using known temperature, Isc, and Voc 
                                popt_rev,pcov_rev = sp.optimize.curve_fit(helper, V_rev, I_rev, p0 = [Rs_guess,Rsh_guess,2], maxfev=100000, bounds=([0,0,1],[np.inf,np.inf,10]))   
                                # report the parameters
                                Rs_rev, Rsh_rev, n_id_rev = popt_rev
                            except:
                                Rs_rev, Rsh_rev, n_id_rev = [np.nan, np.nan, np.nan]
                            # plot the data
                            h1.set_xdata(V_rev)
                            h1.set_ydata(I_rev/active_area*1000)
                            # plot the fit
                            if ~np.isnan(Rs_rev):
                                h2.set_xdata(V_rev)
                                h2.set_ydata(helper(V_rev,Rs_rev,Rsh_rev,n_id_rev)*1000/active_area)
                            else:
                                pass
                            
                            #if fr == 0:
                            ax8[0].plot([0,1.2], [0,0],'k--',linewidth=1)
                            #ax1[0].plot(Vs, JVcurves[fr,:],'.--',markersize=12,linewidth=2)
                            #ax1[0].add_line([0,0], [1.2, 0])
                            ax8[0].set_xlabel('$Voltage\ [V]$')
                            ax8[0].set_ylabel('$Current\ Density\ [mA/cm^2]$')
                            ax8[0].set_ylim([-25, 5])
                            ax8[0].set_xlim([0, 1.2])
        
                        ani8 = animation.FuncAnimation(fig8, animate_JV_PL, frames=num_valid_files)
                        writer = FFMpegWriter(fps=15, metadata=dict(artist='Me'), bitrate=1800)
                        #ani8.save(os.path.join(WriteDir, 'JV_PL_movie_rev_sweep.mp4'), writer=writer)

                # Time stopping t(PLQY-75)
                thres = PLQY_xy0t0[0, 0] * 0.75
                if is_masked:
                    time_stopping = find_plqy_75(plqy_all.data, thres) * const.vidfreq
                    logical_all_mask = np.all(binary_masks.astype(bool), axis=0)
                    time_stopping = np.ma.array(time_stopping, mask=(~logical_all_mask))
                else:
                    time_stopping = find_plqy_75(plqy_all, thres) * const.vidfreq



                plt.figure()
                plt.imshow(time_stopping)
                ax = plt.gca()
                ax.set_yticklabels([])
                ax.set_xticklabels([])

                cb = plt.colorbar()
                cb.set_label('$t(\\mathrm{PLQY}-75)$')
                plt.savefig(os.path.join(WriteDir, 'time_plqy_75.png'), dpi=300, bbox_inches='tight')
                plt.close()

                # Mean PLQY with time
                plqy_mean = np.mean(plqy_all, axis=0)

                plt.figure()
                plt.imshow(plqy_mean)
                ax = plt.gca()
                ax.set_yticklabels([])
                ax.set_xticklabels([])

                cb = plt.colorbar()
                cb.set_label('<PLQY>_t')
                plt.savefig(os.path.join(WriteDir, 'plqy_mean_t.png'), dpi=300, bbox_inches='tight')
                plt.close()

                # Std PLQY with time
                plqy_std = np.std(plqy_all, axis=0)

                plt.figure()
                plt.imshow(plqy_std)
                ax = plt.gca()
                ax.set_yticklabels([])
                ax.set_xticklabels([])

                cb = plt.colorbar()
                cb.set_label('$\\sigma_t$')
                plt.savefig(os.path.join(WriteDir, 'plqy_std_t.png'), dpi=300, bbox_inches='tight')
                plt.close()

                # std/mean PLQY with time
                plqy_cov = np.std(plqy_all, axis=0)

                plt.figure()
                plt.imshow(plqy_cov)
                ax = plt.gca()
                ax.set_yticklabels([])
                ax.set_xticklabels([])

                cb = plt.colorbar()
                cb.set_label('$\sigma$ / $\\langle \\mathrm{PLQY}\\rangle$')
                plt.savefig(os.path.join(WriteDir, 'plqy_cov.png'), dpi=300, bbox_inches='tight')
                plt.close()


                # Fourier time spectrum
                if fft_stats:
                    fft_mean_vid_windowed = np.fft.fftshift(np.fft.fft2(
                        plqy_all * two_d_window[None, ...]), axes=(1, 2))

                    radi_angle_array, size_array, radi_range, angle_range = \
                        generate_radii_angle_array(fft_mean_vid_windowed)

                    mean_fft_spectrum = np.ma.mean(np.ma.log10(radi_angle_array), axis=2).data
                    _until = math.ceil(mean_fft_spectrum.shape[1] / np.sqrt(2))

                    coord_x_max = _until / (pixel_per_um * primary_videos.shape[-1])

                    plt.figure()
                    plt.imshow(mean_fft_spectrum[:, :_until],
                               extent=[0, coord_x_max, t[0].item(), t[-1].item()],
                               aspect='auto', origin='lower')
                    plt.ylabel('time [$s$]')
                    plt.xlabel('reciprocal space $s$ [$\\mu m^{-1}$]')
                    cbar = plt.colorbar()
                    cbar.set_label('$\\log(F)$')
                    plt.savefig(os.path.join(WriteDir, 'fft_time_spectrum.png'),
                                dpi=300, bbox_inches='tight')
                    plt.close()

                # Make a dataframe of features and save
                Incident_flux_ = _inverse_transform(Incident_flux, valid_indices, numfiles)
                QFLS_norm = QFLS_xy0t0/QFLS_xy0t0[0]

                df = pd.DataFrame(np.concatenate([t[:, None], Incident_flux_[:, None]], axis=1),
                                  columns = ['t', 'Incident_flux'])

                feat_list = ['xy0t0',
                             'exp_times',
                             'xy0t0_cts',
                             'PLQY_xy0t0',
                             'xy1t0',
                             'xy2t0',
                             'xy3t0',
                             'xy0t1',
                             'xy0t1Norm',
                             'xy1t1',
                             'xy2t1',
                             'xy3t1',
                             't0xy1',
                             't1xy0',
                             'QFLS_xy0t0',
                             'QFLS_norm',
                             'beta_mean_xy_vs_t',
                             'beta_std_xy_vs_t',
                             'cv_slopes']
                
                #append dark field stats to feat_list
                if dark_field:
                    feat_list.append('DF_means')
                    feat_list.append('DF_medians')
                    feat_list.append('DF_stds')
                    feat_list.append('DF_skews')
                    feat_list.append('DF_kurts')
                    # if the reflectance calibration was found, add specular reflectance stats too
                    if DF_cal_exists:
                        feat_list.append('Rdiff_means')
                        feat_list.append('Rdiff_medians')
                        feat_list.append('Rdiff_stds')
                        feat_list.append('Rdiff_skews')
                        feat_list.append('Rdiff_kurts')     
                    
                # add bright field stats to feat_list
                if bright_field:
                    feat_list.append('BF_means')
                    feat_list.append('BF_medians')
                    feat_list.append('BF_stds')
                    feat_list.append('BF_skews')
                    feat_list.append('BF_kurts')
                    # if the reflectance calibration was found, add specular reflectance stats too
                    if BF_cal_exists:
                        feat_list.append('Rspec_means')
                        feat_list.append('Rspec_medians')
                        feat_list.append('Rspec_stds')
                        feat_list.append('Rspec_skews')
                        feat_list.append('Rspec_kurts')                       
                    
                #TODO: add device characteristics to feat_list
                if sample_type == 'device':
                    if Voc_collected:
                        feat_list.append('Voc_timeseries')
                    if MPPT_collected:
                        feat_list.append('Vmpp_timeseries')
                        feat_list.append('Impp_timeseries')
                    if Isc_collected:
                        feat_list.append('Isc_timeseries')
                    if Light_IV_collected:
                        feat_list.append('Voc_fwd_timeseries')
                        feat_list.append('Isc_fwd_timeseries')
                        feat_list.append('FF_fwd_timeseries')
                        feat_list.append('Rs_fwd_timeseries')
                        feat_list.append('Rsh_fwd_timeseries')
                        feat_list.append('n_id_fwd_timeseries')
                        feat_list.append('I0_fwd_timeseries')
                        feat_list.append('IL_fwd_timeseries')
                        feat_list.append('R2_fwd_timeseries')
                        feat_list.append('Voc_rev_timeseries')
                        feat_list.append('Isc_rev_timeseries')
                        feat_list.append('FF_rev_timeseries')
                        feat_list.append('Rs_rev_timeseries')
                        feat_list.append('Rsh_rev_timeseries')
                        feat_list.append('n_id_rev_timeseries')
                        feat_list.append('I0_rev_timeseries')
                        feat_list.append('IL_rev_timeseries') 
                        feat_list.append('R2_rev_timeseries')
                        feat_list.append('Rs_fwd_timeseries_slope')
                        feat_list.append('Rsh_fwd_timeseries_slope')
                        feat_list.append('Rs_rev_timeseries_slope')
                        feat_list.append('Rsh_rev_timeseries_slope')
                        
                feat_dict = dict()
                for name in feat_list: feat_dict[name] = eval(name)

                def plot_save(name, values, dpi=300, indicies=None):
                    if indicies is None:
                        indicies = np.arange(t.shape[0])
                    plt.figure()
                    span = np.max(np.abs(values))/np.min(np.abs(values))
                    if span > 1000:
                        plt.semilogy(t[indicies], values[indicies], '.', markersize=12)
                    else:
                        plt.plot(t[indicies], values[indicies], '.', markersize=12)
                    plt.xlabel('time [s]')
                    plt.ylabel(name)
                    plt.savefig(os.path.join(WriteDir, name), dpi=dpi, bbox_inches='tight')
                    plt.close()
                    df[name] = values

                def plot_para_save(col1, col2, name, dpi=300):
                    plt.figure()
                    plt.plot(df[col1], df[col2],'.', markersize=12)
                    plt.xlabel(col1)
                    plt.ylabel(col2)
                    plt.savefig(os.path.join(WriteDir, name), dpi=dpi, bbox_inches='tight')
                    plt.close()

                for ii in range(len(feat_list)):
                    name = feat_list[ii]
                    values = feat_dict[name]
                    plot_save(name, values)
                    if numfiles != num_valid_files:
                        plot_save('%s_valid_only' % name, values, indicies=valid_indices)


                frac_bright = np.array([np.mean(beta > 0) for beta in betas])
                plt.figure()
                plt.plot(t[valid_indices], frac_bright, '.--', label='brightening')
                plt.plot(t[valid_indices], 1 - frac_bright, '*:', label='dimming')

                plt.ylabel('Areal fraction [%]')
                plt.xlabel('Time [s]')

                plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.13), framealpha=.0)
                plt.savefig(os.path.join(WriteDir, 'fraction_photo_brightening'), dpi=300, bbox_inches='tight')
                plt.close()

                df['frac_bright'] = _inverse_transform(frac_bright, valid_indices, numfiles)
                df['frame_corrupted'] = (~valid_indices)

                if sample_type == 'film' and experiment_info['Ld_data']:
                    df = pd.concat([df, Ld_df], axis=1)
                    plot_para_save('DC LD [nm]', 'QFLS_xy0t0', 'QFLS-Ld')
                    plot_para_save('DC LD [norm]', 'QFLS_norm', 'QFLS-Ld_norm')
                    if np.isnan(df['t'].iloc[-1]):
                        df['t'] = np.array([experiment_info['vidTimeIntervalMs'] / 1000 * ii
                              for ii in range(df.shape[0])])

                df.to_csv(os.path.join(WriteDir, 'analyzed_data.csv'))

                # Rewrite sample and experimennt metadata to WriteDir
                with open(WriteDir + '/experiment_info.json', 'w') as outfile:
                    json.dump(experiment_info, outfile)
                with open(WriteDir + '/sample_info.json', 'w') as outfile:
                    json.dump(sample_info, outfile)

                # Authomatically push results to Drive
                #source = 'C:/Users/hughadm/Documents/Users/Ryan/hpdb/data/push_to_drive2/analyzed_plva_data/'
                #dest =  'drive:Machine_Learning/Data/Timeseries' OLD RYAN NAME
                #dest =  'gdrive:Machine_Learning/Data/Timeseries' # Effort_Perovskties_2; filled mid-October 2020
                #dest = 'gdrive4:Effort_Perovskites_4/Machine_Learning/Data/Timeseries/' #Links to a folder in Tim Siegler's Drive. To reconfigure a new drive, open cmd and run "rclone config" and follow instructios
                #command = (['rclone', 'copy', source, dest])
                #subprocess.Popen(command)


if __name__ == '__main__':
    main()
